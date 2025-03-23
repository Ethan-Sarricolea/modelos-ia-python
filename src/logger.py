import logging

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Aplicaci�n iniciada")
logging.error("Ocurri� un error")
