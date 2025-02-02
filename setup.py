from setuptools import setup, find_packages

setup(
    name="glove_kob_model",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "glove_model": ["model/*"],
    },
    install_requires=[
        "numpy",
        "scikit-learn",  # Для примеров использования
    ],
    description="GloVe word embeddings for Russian language",
    author="audetv",
    author_email="audetv@gmail.com",
    url="https://github.com/audetv/glove_kob_model",
)