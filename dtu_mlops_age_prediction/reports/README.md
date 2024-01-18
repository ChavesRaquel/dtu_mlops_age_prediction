---
layout: default
nav_exclude: true
---

## Title of the project: Facial Age Prediction**

*Overview*

This project aims to create an age prediction model for facial images, enabling applications such as personalized content recommendations and demographic analysis. Leveraging the PyTorch Image Models (https://github.com/huggingface/pytorch-image-models), a versatile and scalable machine learning framework, we intend to develop a high-accuracy model for estimating the age of individuals.

*Framework Integration*

To seamlessly integrate the PyTorch Image Models into our project, we'll establish a virtual environment using conda. This ensures project isolation and efficient dependency management. Detailed instructions for reproducing the environment will be available in the project documentation.

*Data*

The initial dataset, Facial Age, comprises diverse facial images annotated with age labels. This curated dataset represents a broad spectrum of age groups, ethnicities, and gender identities. Preprocessing steps will involve extracting facial features and annotations to enhance model training.

*Dataset Source*

The dataset is sourced from https://www.kaggle.com/datasets/frabbisw/facial-age. Its richness and diversity will enable the model to generalize well across various facial characteristics.

*Models*

Our approach involves exploring various machine learning models for age prediction, including Convolutional Neural Networks (CNNs), Support Vector Machines (SVMs), and Ensemble Models. These models will undergo evaluation based on performance metrics, with iterations on the chosen architecture to achieve optimal results.

*Project Documentation*

The project's comprehensive documentation will include details on model architectures, training procedures, and performance evaluations. This README serves as a concise guide to understanding our goals, methodologies, and the resources involved.

By integrating the PyTorch Image Models framework and dataset seamlessly, this project ensures easy reproducibility and experimentation. We invite contributions and collaborations to enhance the accuracy and robustness of the age prediction model.


Project done for the course Machine Learning Operations (02476) from Technical University of Denmark (DTU) by:
- Alba Castrillo Perote (s230221)
- Raquel Chaves Martinez (s231844)
- Jorge Martinez Requena (s211980)
- Victor Luna Santos (s222931)

-------- 

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [ ] Create a git repository
* [ ] Make sure that all team members have write access to the github repository
* [ ] Create a dedicated environment for you project to keep track of your packages
* [ ] Create the initial file structure using cookiecutter
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [ ] Add a model file and a training script and get that running
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [ ] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [ ] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [ ] Get some continuous integration running on the github repository
* [ ] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer: 

---Group 39---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer: 

---*s230221, s231844, s211980, s222931*---

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We are using the Pytorch Image Models framework (timm), given that this project consists in predicting the ages of people from a face picture, in the form of classifier from 1 to 90+ years old.
From the available models in this framework, we used resnet18, widely known for image classification and a good base model in order to optimise training for the case. This framework enabled the project with an improved accuracy than the accuracy that could have been obtained from a model from scratch and a higher decrease in the loss parameter. It improved the training process, but such a complex classification task, still has not very good accuracy. ---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

--- Conda was used to create a virtual environment, named dtu_mlops_age_prediction. The first installation and the process of adding new packages to the environment has been done by calling pip. Every required package in the project has been gathered in a requirements.txt file the same way as it was showed in the mlops course, and it is updated every time the packages change. In order for a new member to build the environment to work in the project, the process should be to clone the repository, create a new conda environment, and run the command *make requirements* or *pip install -r requirements.txt* in the main folder of the repository and all the packages and instances should be installed correctly.--- 

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- question 5 fill here ---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- We have applied pep8 to our code. Having a standard codebase maked the code more readable. When code is easier to understand by many people, it gives access to an easier collaboration between people in big groups that code differently and also eases maintenance and debugging, especially in large projects as many lines of code are involved. ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- *Test data:* We test that the label and the images of the dataset have been correctly generated and they have the same length. 
*Test model:* We test that the model has the correct architecture, and with an input of a tensor the same size of our images in the dataset has the desired output.---

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- From the coverage report, we extract that the total coverage of the code is 100%, which means that all 57 statements across the analyzed files are covered by tests. However, this code coverage doesn't mean that the code is completely error-free as the tests that we have managed to implement are quite basic. An example of this is the test of the data, where we simply check if the train and test data and labels have been correctly generated. Then, it is more related to the quality of the tests implemented than to their coverage. Some measures that can be taken to improve the quality of the code are to implement other testing practices. ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- Our workflow included branches. We used different branches to work collaboratively in different parts of the project without affecting other parts of the project, for example, one branch was used for the first training stages, and other branch was used to set up the dvc. As most of us worked a bit in every part, the project was set so that each of us worked in different branches everytime but at the same time we could change the parts of the project we were working on. Never at the same time in the same branch. As long as some part worked, we used merge to main. Parts of the project that required previous parts were branched from main once the previous were working and were merged to main. We have not used pull request because we have worked all together physically and we communicated. ---


### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- We have followed the necessary steps to implement DVC. Our data has been stored in a remote location, although we encountered a setback in terms of pulling to access the data.  Initially, we integrated DVC with our personal Google Drive to store our data. However, a significant drawback of this approach is that we are required to authenticate each time we attempt to either push or pull the data. Consequently, we opted to used an API provided by GCP instead. We created a bucket through the GCP platform and subsequently migrated our data storage from our Google Drive to this new Google Cloud Storage, ultimately pushing the data to the cloud.

The implementation of version control might be very helpful as it can facilitate a clear understanding of how the data evolves, it can enable all the team members to work on the data simultaneously without encountering conflicts or data loss, it also allows for easy reproduction ensuring that every experiment is conducted with the same data, provides a safety net in the event of data loss or corruption, and permits the tracking of different versions of the data and their corresponding results, thereby enabling to select the optimal outcome. ---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- question 11 fill here ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- We used various things, a config file with hydra, to run different trainings with config_train.yaml, where learning rate, number of epochs and batch size is set. But mainly a config file with wandb sweeper to run many experiments iterating through the same parameters to find the optimal combination of them, these parameters have been delimited in wandb_sweep.yaml . Both files have been set in the config folder  ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- We combine hydra with wandb, with wandb sweep we can see different combinations of parameters and their output when training a model. From that information, we can edit the config file that hydra uses to get the parameters that we want to have in order to have an experiment with the desired accuracy and loss. With this, we can save a model that has been trained with those parameters using the hydra config file and that model can be imported to pass information through. With this, we ensure that we have a log of all the experiments and their outputs including having the ability to reproduce the model that gave those results adn then work from that ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- For our project we created an image for each of the files that we created. In other words, one docker images to download the data, one for creating the dataset, one for training the model and finally one to make predictions using the model. To run each of the docker images,  it's simply required to do: 'docker run trainer:latest' for training the model; 'docker run  redicter:latest' for making predictions; 'docker run download:latest' for downloading the data from kaggle and 'docker run dataset:latest' for creating the dataset. ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- Debugging was done while coding. A lot of problems have appeared during the coding and those bugs were solved while they were appearing. For this code, a single profiling has been used in order to see using tensorboard locally with pytorch profiler, how the different parts of the training affect the computation time in the execution of the code. We have observed the data, but due to time reasons, we couldn't apply debugging methods to improve performance of the project---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- [this figure](figures/bucket_1.jpg) [this figure](figures/bucket_2.jpg) ---
### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- [this figure](figures/registry.jpg) ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- [this figure](figures/cloud_build.jpg) ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- The deployment has been done locally, we've used the packages fastapi and uvicorn. In the deployment we pass an image into the model and it outputs the predicted age. The aplication is done via Fastapi and the paramenters are entered via decorators, First a root where the status of the HTTP is displayed and in the predict path, the input and output are given. Uvicorn is run to host it in a localhost. The app is intuitive with buttons. Cloud deployment has not been done given that a correct cloud setup to do it has't been reached. ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- We have not been able to implement monitoring due to appearance of crucial problems and time limitations. Monitoring is important because it enables tracking the performance in a real-time basis, this makes it easier to detect and solve problems when appearing before the problem is too big to handle. Consistency in performance is achieved by an effective monitoring, also, reliability is very important in order to make a successful application, thus longevity. Also some more in depth monitoring can enable further development and an easier scalability of the application, ensuring as well longevity as the application would be updated to the standards of the moment.---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- s230221 used 0,51$ and the service costing the most was cloud storage. s231844 used..... s211980 used 45$ and the service consting the most was compute engine. s222931 used 5,4$ ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---
