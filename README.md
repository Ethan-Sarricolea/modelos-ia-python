# My Project name

## Descripción
Este es un sistema base para proyectos en Python, con una estructura organizada para facilitar el desarrollo y mantenimiento.

## Estructura del Proyecto

```
Project/
│── assets/     # Recursos estaticos
│── config/     # Configuracion
│    ├── data.json
│── docs/       # Documentos
├── notes       # Notas jupyter
|    ├──libreta.ipynb
│── logs/       # Logs (Registro de ejecuciones)
│── src/        # Codigo fuente
|    ├── module
│       ├── main.py
│       ├── logger.py
│       ├── utils.py
│── tests/      # Pruebas automatizadas
│    ├── test_main.py
│── .gitignore
│── LICENSE
│── README.md
│── requirements.txt
```

## Librerias utilizadas

- Pandas: Para leer, manipular y analizar datos
- NumPy: Para operaciones mas rapidas
- scikit-learn: Con herramientas para modelos estadisticos y de machine learning como: **clasificación, regresion, agrupamiento, etc.**
- matplotlib: Crear graficos y visualizar datos
- seaborn: Crear graficos y visualizar datos
- Tensorflow o Pytorch: Para deep learning y redes neuronales.

Para ver la lista de liberias instaladas
```bash
pip list
```

## Instalación

```sh
git clone <URL_DEL_REPOSITORIO>
cd nodrisa_system
python -m venv venv
source venv/bin/activate  # En Windows: venv/Scripts/activate
pip install -r requirements.txt
```

## Ejecución

```sh
python src/main.py

```

## Contribución
Si deseas contribuir a este proyecto, por favor crea un fork y envía un pull request.

# Notas de curso
Las notas del curso se encuentran en mi repositorio de notas academicas, por lo que posteriormente las pasare aqui.

## Notebooks

En la carpeta notebooks se encuentran los aprendizajes y modelos.
