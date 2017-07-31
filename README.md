# Center for Medical Innovation, Software, and Technology
### Sidra Medical Research Center
![](https://raw.githubusercontent.com/CMIST/HiringExercise_SeniorMLEngineer/master/logo_cmist.png "CMIST LOGO")

Dear Candidate,

Congratulations on your candidacy to the CMIST Division (Center for Medical Innovation, Software, and Technology) at Sidra Medical Research Center.  Our hiring process is gated and involves four steps:
1.	**Technical Competency** - evaluated from job application
2.	**Technical Mastery** - evaluated via application and phone screen with hiring manager
3.	**Delivery Capability** - evaluated via this take-home exercise
4.	**Organizational Fit** - evaluated via group interview 

This repository details item three – a take-home exercise which aims to evaluate your ability to deliver.  With this, we are observing three major things:
1.  Ability to find and use existing solutions
2.	Technical Correctness
3.	Communication and Presentation
4.	Self-Management

### Rationale for this Exercise
Without working with confidential medical images, this exercise closely resembles types of work we do in the group, specifically:
* Find existing solutions we can build upon, try to not "re-create the wheel"
* Dealing with heterogeneous in-the-wild samples (various sizes, various aspect ratios.)  
* Execute classification and/or object detection with 0,1,1+ cases per image
* Communicating ideas
* Managing own efforts 

This exercise is geared towards Position # 4286 (Senior Developer: Machine Learning Computer Vision,) which is a complementary role to Position # 4338 (Machine Learning: Deep Neural Networks Specialist) and Position # 4285 (Developer: Machine Learning Computer Vision.)  Since this is a complementary role, we tried to make the hiring exercise realistic, even to the point of interacting with the complementary roles' hiring exercises.  Accordingly, you may wish to see the hiring exercise for the other roles (https://github.com/CMIST/) so you can learn from or build upon those -- there are several great solutions available publicly on GitHub from other candidates.

### The Exercise

Your first goal is to create a desktop application which runs live inference on video streams looking for any object of your choice (chairs, books, cups, cats, whatever.)  The live video stream should be displayed and the object(s) detected should be noted on the application (either in logs, consoles, or directly in the UI.)  This is a really easy exercise with hundreds of available solutions when working with single images -- your task is to extend this to a video solution.  Even the video solution has a number of demonstrated apps that do this:
* https://github.com/richardstechnotes/rtndf/blob/master/Python/imageproc/imageproc.py
* https://richardstechnotes.wordpress.com/2016/07/25/processing-video-streams-with-tensorflow-and-inception-v3/

The second part of the exercise will offers you a choice of one of two tasks:
1. Increase the inference throughput of your solution above and note various things you tried and how they affected the throughput
2. Swap in a different model, instead of using a standard trained model, use something that you trained on an object of your own choice (e.g., rosary beads.) You will want to also keep some negative and positive images.

![](https://raw.githubusercontent.com/saifrahmed/HiringExercise_SeniorMLEngineer/master/sugeknight.png "Suge Knight")

Dont worry about making it pretty, the applicaiton can be ugly.  You are being judged on functionality and reproduceability, not UI/UX/design.  Please deliver back all scripts (and everything else referenced) as well as a corresponding set of training images you used.  Your scripts should be all-encompassing, including pipeline steps and acquisition of any training sets.  If you need a setup script, that can also be included.

Your submission should be via Github.  The exercise should be done in Python or Java.  Please do not use paid platforms like MATLAB.  For the deep learning portion, you can use Theano, DL4J, TensorfFlow, TensorFlow + Keras or Caffe2 (our in-house standard is TensorFlow.)  To this point, please detail your selected environment, machine learning / computer vision platform and any other setup steps so we can re-train and re-test your model from scratch.

There is no need to re-create the wheel here, use any and all open-source or vendor-provided scripts that you can.  Ensure to include them so the submission is all-encompassing.  **Do not** include anything irrelevant to the goal of this exercise, we want the minimal set of files/work which achieves the goal and nothing more.

### Discussion Questions
1.	What foundation did you use for your project?  What did you modify and what did you keep identical?
2.	Describe your processing pipeline and pre-processing steps, if any
3.	How much time did you spend on this exercise?
4.  What shortcuts did you have to take to deliver the exercise in the short period of time allotted?
5.	If you had more time, how would you expand on this submission?  What is your wishlist for the app?

Good Luck!