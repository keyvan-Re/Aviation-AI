import os

if __name__ == '__main__':
    # این دستور به پارس‌پک می‌گوید که فایل اصلی ما (main.py) را با Streamlit اجرا کند
   
    os.system("streamlit run main.py --server.port=8080 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false")

    

