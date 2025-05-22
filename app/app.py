import streamlit as st
import pandas as pd
import joblib
import os.path

# Загрузка модели
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'GBC_model.pkl'))

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Возникли трудности с загрузкой модели")
    st.stop()

st.title("Предсказание на основе CSV")
 
uploaded_file = st.file_uploader("Загрузите ваш CSV-файл с информацией клиентов банка", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Небольшая подготовка данных
        df = (df.set_index('id')
                .drop(['CustomerId', 'Surname'], axis=1))

        st.write("Пример входных данных:")
        st.dataframe(df.head())

        y_pred = model.predict(df)
        df["Exited"] = y_pred

        st.success("Предсказания выполнены!")
        st.write("Предсказанные данные:")
        st.dataframe(df.head())

        # Сохранение в память
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Скачать результат (CSV)",
            data=csv_out,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Возникла ошибка при обработке файла: {e}")
