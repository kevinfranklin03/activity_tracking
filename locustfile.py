# from locust import HttpUser, task, between

# class EndpointUser(HttpUser):
#     wait_time = between(1, 3)  # Simulate users with random wait times between requests

#     @task
#     def predict(self):
#         # Replace with your Azure ML endpoint URL
#         url = "https://coursework-kcdvm.northeurope.inference.ml.azure.com/score"
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": "Bearer TiHpzA7sX7Iee4jRDR4SuZghsqMN3k3g"  # Replace with your endpoint key
#         }
#         # Replace this with input data matching your model's requirements
#         data = {
#             "data": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
#         }

#         response = self.client.post(url, json=data, headers=headers)
#         print(response.status_code, response.text)

from locust import HttpUser, task

class ModelUser(HttpUser):
    @task
    def make_inference(self):
        payload = {"data": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]}  # Adjust as per your input format
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer TiHpzA7sX7Iee4jRDR4SuZghsqMN3k3g"  # Replace with your Azure API key
        }
        self.client.post("https://coursework-kcdvm.northeurope.inference.ml.azure.com/score", json=payload, headers=headers)
