library(shiny)
library(quantmod)
library(xgboost)
library(caret)
library(TTR)
library(dplyr)

ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "cosmo"),
  
  tags$head(tags$style(HTML("
    body {
      background: linear-gradient(to right, #2c3e50, #3498db);
      font-family: 'Arial', sans-serif;
      color: black;
    }
    .title {
      text-align: center;
      font-size: 32px;
      font-weight: bold;
      color: white;
      padding: 15px;
      background: rgba(0, 0, 0, 0.2);
      border-radius: 10px;
    }
    .card-box {
      background: rgba(255, 255, 255, 0.95);
      color: black;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .btn-predict {
      background: linear-gradient(to right, #3498db, #2ecc71);
      color: white;
      font-size: 16px;
      padding: 12px 25px;
      border-radius: 8px;
      border: none;
      transition: all 1s ease-in-out;
    }
    .btn-predict:hover {
      background: linear-gradient(to right, #2ecc71, #3498db);
      transform: scale(1.1);
    }
    h4 {
      color: black;
    }
  "))),
  
  div(class = "title", "AI Stock Price Predictor"),
  
  fluidRow(
    column(4,
           div(class = "card-box",
               textInput("stock_symbol", "Enter Stock Symbol:", "AAPL"),
               dateRangeInput("date_range", "Select Date Range:",
                              start = "2020-01-01", end = Sys.Date()),
               numericInput("days_ahead", "Days Ahead to Predict:", 3, min = 1, max = 10),
               downloadButton("download_data", "Download Prediction", class = "btn btn-dark"),
               actionButton("predict_btn", "Predict Price", class = "btn-predict")
           )
    ),
    column(8,
           div(class = "card-box",
               h4("Prediction Result"),
               verbatimTextOutput("prediction")
           ),
           div(style = "margin-bottom: 20px;"),
           div(class = "card-box",
               h4("Stock Price Chart"),
               plotOutput("stock_plot")
           ),
           div(class = "card-box",
               h4("Model Performance (Test RMSE)"),
               verbatimTextOutput("rmse_output")
           )
    )
  )
)

server <- function(input, output) {
  prediction_data <- reactiveVal(NULL)
  
  observeEvent(input$predict_btn, {
    stock_symbol <- toupper(input$stock_symbol)
    start_date <- input$date_range[1]
    end_date <- input$date_range[2]
    days_ahead <- input$days_ahead
    
    tryCatch({
      # Pull stock data
      stock_xts <- getSymbols(stock_symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
      
      df <- data.frame(Date = index(stock_xts), coredata(stock_xts))
      colnames(df) <- c("Date", "Open", "High", "Low", "Close", "Volume", "Adjusted")
      df <- na.omit(df)
      
      # Features
      df$Lag1_Close <- dplyr::lag(df$Close, 1)
      df$Lag2_Close <- dplyr::lag(df$Close, 2)
      df$Lag3_Close <- dplyr::lag(df$Close, 3)
      df$RSI <- RSI(df$Close)
      df$MACD <- MACD(df$Close)[, "macd"]
      df <- na.omit(df)
      
      set.seed(123)
      train_index <- createDataPartition(df$Close, p = 0.8, list = FALSE)
      train_data <- df[train_index, ]
      test_data <- df[-train_index, ]
      
      features <- c("Lag1_Close", "Lag2_Close", "Lag3_Close", "RSI", "MACD")
      preProc <- preProcess(train_data[, features], method = c("center", "scale"))
      train_scaled <- predict(preProc, train_data[, features])
      test_scaled <- predict(preProc, test_data[, features])
      
      dtrain <- xgb.DMatrix(data = as.matrix(train_scaled), label = train_data$Close)
      dtest <- xgb.DMatrix(data = as.matrix(test_scaled), label = test_data$Close)
      
      model <- xgboost(data = dtrain, objective = "reg:squarederror",
                       nrounds = 100, max_depth = 6, eta = 0.1, verbose = 0)
      
      predictions <- predict(model, dtest)
      rmse <- sqrt(mean((predictions - test_data$Close)^2))
      
      # Multi-day future prediction
      last_row <- tail(df, 1)
      future_preds <- numeric(days_ahead)
      
      for (i in 1:days_ahead) {
        input_features <- data.frame(
          Lag1_Close = last_row$Lag1_Close,
          Lag2_Close = last_row$Lag2_Close,
          Lag3_Close = last_row$Lag3_Close,
          RSI = last_row$RSI,
          MACD = last_row$MACD
        )
        
        input_scaled <- predict(preProc, input_features)
        next_pred <- predict(model, xgb.DMatrix(as.matrix(input_scaled)))
        
        future_preds[i] <- next_pred
        
        # Update last_row for next prediction
        last_row$Lag3_Close <- last_row$Lag2_Close
        last_row$Lag2_Close <- last_row$Lag1_Close
        last_row$Lag1_Close <- next_pred
        # Keep RSI and MACD constant (approximation)
      }
      
      output$prediction <- renderText({
        paste0("Predicted closing prices for next ", days_ahead, " day(s):\n",
               paste0("Day ", 1:days_ahead, ": $", round(future_preds, 2), collapse = "\n"))
      })
      
      output$rmse_output <- renderText({
        paste("Test RMSE:", round(rmse, 4))
      })
      
      output$stock_plot <- renderPlot({
        plot(df$Date, df$Close, type = "l", col = "blue", xlab = "Date", ylab = "Close Price",
             main = paste("Historical Prices of", stock_symbol))
        points(rep(tail(df$Date, 1), days_ahead), future_preds, col = "red", pch = 19)
        legend("topleft", legend = c("Actual Prices", "Predicted Prices"),
               col = c("blue", "red"), lty = 1, pch = c(NA, 19))
      })
      
      prediction_data(data.frame(Day = 1:days_ahead, Predicted_Close = round(future_preds, 2)))
      
    }, error = function(e) {
      output$prediction <- renderText({
        paste("Error: Unable to fetch data for", stock_symbol, "\n", e$message)
      })
    })
  })
  
  output$download_data <- downloadHandler(
    filename = function() {
      paste0("prediction_", input$stock_symbol, ".csv")
    },
    content = function(file) {
      write.csv(prediction_data(), file, row.names = FALSE)
    }
  )
}

shinyApp(ui = ui, server = server)