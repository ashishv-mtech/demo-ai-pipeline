To Build a production Grade ML Model pipeline which is served as FastAPI Containerized APP on Cloud with Git and Automated Github Testing 



Flow:
1. Dataset: 
    - Clean the Dataset in Notebook , Prepare for ML Model training

2. Training:
    - train the Model with Experimental Tracing/Record Keeping using MLFLOW 
    - Evaluation with proper metrics

3. Backend:
    - Wrap the Entire thing in FastApi backend 
    - Containerize it using Docker 

4. Tests and Scripts : 
    - Add tess of : Data Validation of User Inputs, APIs, Training , etc. using Pytest
    - Add scripts of training, config etc.


5. Github WorkFlow
    - Add Automated script to Test in yml files


6. Deploy to Cloud using simple way 