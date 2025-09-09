from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import StockPredictionSerializer
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode for servers
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
import os
from django.conf import settings

# Utility function to save plot
def save_plot(filename):
    filepath = os.path.join(settings.MEDIA_ROOT, filename)
    plt.savefig(filepath)
    plt.close()
    return f"{settings.MEDIA_URL}{filename}"

class StockPredictionAPIView(APIView):
    def post(self, request):
        serializer = StockPredictionSerializer(data=request.data)
        if serializer.is_valid():
            ticker = serializer.validated_data['ticker']

            try:
                # Fetch historical data
                now = datetime.now()
                start = datetime(now.year - 10, now.month, now.day)
                df = yf.download(ticker, start=start, end=now)

                if df.empty or len(df) < 200:
                    return Response(
                        {"error": "Not enough historical data for this ticker."},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                df = df.reset_index()

                # Plot Closing Price
                plt.figure(figsize=(12, 5))
                plt.plot(df['Close'], label='Closing Price')
                plt.title(f'Closing Price of {ticker}')
                plt.xlabel('Days')
                plt.ylabel('Price')
                plt.legend()
                plot_img = save_plot(f"{ticker}_plot.png")

                # 100-day Moving Average
                ma100 = df['Close'].rolling(100).mean()
                plt.figure(figsize=(12, 5))
                plt.plot(df['Close'], label='Closing Price')
                plt.plot(ma100, 'r', label='100 DMA')
                plt.title(f'100-Day Moving Average of {ticker}')
                plt.xlabel('Days')
                plt.ylabel('Price')
                plt.legend()
                plot_100_dma = save_plot(f"{ticker}_100_dma.png")

                # 200-day Moving Average
                ma200 = df['Close'].rolling(200).mean()
                plt.figure(figsize=(12, 5))
                plt.plot(df['Close'], label='Closing Price')
                plt.plot(ma100, 'r', label='100 DMA')
                plt.plot(ma200, 'g', label='200 DMA')
                plt.title(f'200-Day Moving Average of {ticker}')
                plt.xlabel('Days')
                plt.ylabel('Price')
                plt.legend()
                plot_200_dma = save_plot(f"{ticker}_200_dma.png")

                # Prepare Data for ML Model
                data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
                data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):])

                scaler = MinMaxScaler(feature_range=(0, 1))
                past_100_days = data_training.tail(100)
                final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
                input_data = scaler.fit_transform(final_df)

                x_test, y_test = [], []
                for i in range(100, input_data.shape[0]):
                    x_test.append(input_data[i-100:i])
                    y_test.append(input_data[i, 0])
                x_test, y_test = np.array(x_test), np.array(y_test)

                # Load model safely
                model_path = os.path.join(settings.BASE_DIR, 'stock_prediction_model.keras')
                if not os.path.exists(model_path):
                    return Response(
                        {"error": "ML model file not found."},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

                model = load_model(model_path)

                # Predictions
                y_predicted = model.predict(x_test)
                y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1)).flatten()
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

                # Final Prediction Plot
                plt.figure(figsize=(12, 5))
                plt.plot(y_test, 'b', label='Original Price')
                plt.plot(y_predicted, 'r', label='Predicted Price')
                plt.title(f'Final Prediction for {ticker}')
                plt.xlabel('Days')
                plt.ylabel('Price')
                plt.legend()
                plot_prediction = save_plot(f"{ticker}_final_prediction.png")

                # Model Evaluation
                mse = mean_squared_error(y_test, y_predicted)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_predicted)

                return Response({
                    "status": "success",
                    "plot_img": plot_img,
                    "plot_100_dma": plot_100_dma,
                    "plot_200_dma": plot_200_dma,
                    "plot_prediction": plot_prediction,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2
                })

            except Exception as e:
                return Response(
                    {"error": f"Server error: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
