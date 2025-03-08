# app.py
from flask import Flask, request, jsonify
import requests
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer


app = Flask(__name__)


# Replace with your TMDb API key
TMDB_API_KEY = 'cd1baff5814003fe88a8d0381b925179'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'


from routes import *