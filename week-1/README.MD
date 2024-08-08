# Minikube Deployment

## 1. Start Minikube

```
minikube start
```

## 2. Start Minikube

```
minikube image load sample:v1
```

## 3. Apply YAML Files

```
kubectl apply -f pod.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f job.yaml
```

## 4. Status Check 

```
kubectl get pods
kubectl get deployments
kubectl get services
kubectl get jobs
```

## 5. Access the Service 

```
minikube service sample-service
```

## 6. Result 

![Alt text](<assets/Screenshot 2024-08-04 at 4.03.11 PM.png>)