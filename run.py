from app import app

if __name__ == '__main__':
    app.run(debug=True)

# with app.test_client() as client:
#     response = client.post('/v1/recommend', json={
#         "duration": 120,
#         "genres": ["Action", "Adventure"]
#     })
#     print(response.data)

