# YOLOv10 : Real Time Detection
![alt text](images/image1.png)
### <span style="color:yellow; font-weight:bold">YOLOv10</span> introduces a new approach to post-processing techniques in object detection methods. It significantly reduces the latency of the model.

# What is Latency ?
## <span style="color:yellow"> Latency refers to the time it takes for the model to analyze an image and tell you what objects it found. The faster the response, the lower the latency, and the smoother the experience. It’s typically measured in milliseconds (ms). 

Refer this link for Evaluation paramter invloved in Object detection models [link](https://medium.com/@nikitamalviya/evaluation-of-object-detection-models-flops-fps-latency-params-size-memory-storage-map-8dc9c7763cfe)


# Methodology
## <span style="color:yellow"> Consistent Dual Assignments for NMS-free Training
![alt text](images/image2.png)

## To combine the strengths of both label assignment strategies, <br> they introduced a one-to-one matching head to the YOLO model alongside the traditional one-to-many head. <br> Both heads are optimized during training, but only the one-to-one head is used for inference, eliminating the need for NMS post-processing and maintaining efficient deployment. **This dual approach enhances accuracy and convergence speed.**