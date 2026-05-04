FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .


# RUN pip install --no-cache-dir -r fastapi
# RUN pip install --no-cache-dir -r fastai
# RUN pip install --no-cache-dir -r joblib
# RUN pip install --no-cache-dir -r uvicorn
# RUN pip install --no-cache-dir -r torch
# RUN pip install --no-cache-dir -r torchvision
# RUN pip install --no-cache-dir -r numpy<2
# RUN pip install --no-cache-dir -r ipython
# RUN pip install --no-cache-dir -r scikit-learn==1.6.1
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]