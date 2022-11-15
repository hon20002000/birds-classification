# birds-classification
based on kaggle BIRDS 450 SPECIES- IMAGE CLASSIFICATION

<b>GRANDALA</b>
![alt text](https://github.com/hon20002000/birds-classification/blob/main/demo_images/GRANDALA.png "GRANDALA")

<b>ABBOTTS BABBLER</b>
![alt text](https://github.com/hon20002000/birds-classification/blob/main/demo_images/ABBOTTS%20BABBLER.png "ABBOTTS%20BABBLER")

# docker-flask-postgres

This is a project about docker-flask-postgres-image classification. To run this on your computer you must first install [docker](https://docs.docker.com/engine/installation/).

## Running

The project is about 110 MB, I upload it at google drive, you can download it at this link [https://drive.google.com/drive/folders/1-oq7-q45IX-ThQvTvYjrUjd4levQ3r5b?usp=sharing], or 




And then run this command.

```
docker-compose up --build -d   # Run the container.
```

The site will be available to you at `localhost:5000` or `your IP:5000`.

As long as your computer builds this image and runs the container, You can browse the site by accessing the site from your mobile phone or other device(under the same wifi), enjoy it!

Here is a demo:
![alt text](https://github.com/hon20002000/birds-classification/blob/main/demo_images/index.png "index")

After prediction, you can click the More Info button to search for more information about this bird.
![alt text](https://github.com/hon20002000/birds-classification/blob/main/demo_images/info.png "info")

<b>Cautions!</b>
This model only has good accuracy in predicting bird photos, since it has only 450 categories, if you upload out of categories photos, it will give funny results, it will find the most similar one out of 450 species of birds ! Although it is possible to additionally train a Yes/No binary model to judge whether it is a bird/non-bird photos, the results are more interesting now, so I leave it. Here are some interesting results.

![alt text](https://github.com/hon20002000/birds-classification/blob/main/demo_images/result1.png "result1")
![alt text](https://github.com/hon20002000/birds-classification/blob/main/demo_images/result2.png "result2")

## How it works

I used the python:3.8 image for construction, and then installed the necessary libraries through Dockerfile, such as flask, torch, psycopg2 and other necessary libraries. The role of flask is to allow users to access index.html, and users can access index.html and Upload a picture, which must be a bird picture, and then the model predicts the name of the bird in the backend and displays the result on the page. The role of postgres here is to save the picture path and the prediction to the database.

## Dataset

This application is to determine the category of bird image uploaded by users. Based on kaggle's BIRDS 450 SPECIES- IMAGE CLASSIFICATION Dataset [https://www.kaggle.com/datasets/gpiosenka/100-bird-species] ~ 2 GB, this dataset is not balanced, so f1_score is used as the metric.


## Model Architecture

`Target model` is based on the ConvNext tiny model ~ 110 MB without pre-training, and knowledge distillation is used to improve the training effect.

`Teacher model`: Pre-trained ConvNext xlarge model ~ 1,300 MB, using nn.CrossEntropyLoss() to calculate loss, no Image augmentation due to long training time (>18h), 5 rounds of training to get f1_score = 0.938, according to the trend of the training data, the F1 score should be improved even higher.

## Loss & Metrics

Classification Loss - nn.CrossEntropyLoss()
Classification metric - f1_score

## Dockerfile

- a. docker-compose for full architecture: synchronous projects
- b. client: HTML Frontend
- c. model: teacher model: transfer learning by ConvNext-xlarge model, student model: trained from scratched ConvNext-tiny model.

## Summary

- I use the xlarge model to teach the tiny model, the f1_score on testing set is about 0.928, the performance dropped by 1%

- the size of xlarge model is 1,300 MB, and the size of tiny model is 110 MB, Effectively reduces the size by 91.5%
