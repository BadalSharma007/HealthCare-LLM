this project is to create a platform where user can consult and

hugging face token : stored in Modal secret (huggingface-secret) — never commit tokens here


Created a new secret 'huggingface-secret' with the key 'HF_TOKEN'

Use it in your Modal app:

                                                                      
@app.function(secrets=[modal.Secret.from_name("huggingface-secret")]) 
def some_function():                                                  
    os.getenv("HF_TOKEN")                                             
                             
Modal deployed yrl:BASE_URL = "https://dev-30--medgemma-medical-medicalmodel-api.modal.run"