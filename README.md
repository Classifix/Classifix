# Streamlit-based Web Application

#### EXPLORE Data Science Academy Classification Predict

## 1) Overview

![Streamlit](resources/imgs/streamlit.png)

This repository forms the basis of _Task 2_ for the **Classification Predict** within EDSA's Data Science course. It hosts template code which will enable students to deploy a basic [Streamlit](https://www.streamlit.io/) web application.

As part of the predict, students are expected to expand on this base template; increasing the number of available models, user data exploration capabilities, and general Streamlit functionality.

#### 1.1) What is Streamlit?

[![What is an API](resources/imgs/what-is-streamlit.png)](https://youtu.be/R2nr1uZ8ffc?list=PLgkF0qak9G49QlteBtxUIPapT8TzfPuB8)

If you've ever had the misfortune of having to deploy a model as an API (as was required in the Regression Sprint), you'd know that to even get basic functionality can be a tricky ordeal. Extending this framework even further to act as a web server with dynamic visuals, multiple responsive pages, and robust deployment of your models... can be a nightmare. That's where Streamlit comes along to save the day! :star:

In its own words:

> Streamlit ... is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours! All in pure Python. All for free.

> It’s a simple and powerful app model that lets you build rich UIs incredibly quickly.

Streamlit takes away much of the background work needed in order to get a platform which can deploy your models to clients and end users. Meaning that you get to focus on the important stuff (related to the data), and can largely ignore the rest. This will allow you to become a lot more productive.

##### Description of files

For this repository, we are only concerned with a single file:

| File Name     | Description                       |
| :------------ | :-------------------------------- |
| `base_app.py` | Streamlit application definition. |

## 2) Usage Instructions

#### 2.1) Cloning a copy of this repo

| :zap: WARNING :zap:                                                                                     |
| :------------------------------------------------------------------------------------------------------ |
| Do **NOT** _clone_ this repository. Instead follow the instructions in this section to _fork_ the repo. |

As described within the Predict instructions for the Classification Sprint, this code represents a _template_ from which to extend your own work. As such, in order to modify the template, you will need to **[fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)** this repository. Failing to do this will lead to complications when trying to work on the web application remotely.

![Fork Repo](resources/imgs/fork-repo.png)

To fork the repo, simply ensure that you are logged into your GitHub account, and then click on the 'fork' button at the top of this page as indicated within the figure above.

#### 2.2) Installing Requirements Packages from requirement file

We recommend setting up a running instance on your own local machine.

To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

1.  Clone the _forked_ repo to your local machine.

```bash
git clone https://github.com/1272371/DN3_Classifiers_Model.git
```

2.  Ensure that you have the prerequisite Python libraries installed on your local machine:

```bash
pip install -r requirments.txt
```

3.  Open Jupyter Notebook from this directory.

```bash
jupyter notebook
```

If the web server was able to initialise successfully, the following message should be displayed within your bash/terminal session:

```
To access the notebook, open this file in a browser:
        file:///C:/Users/mncub/AppData/Roaming/jupyter/runtime/nbserver-12516-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=657a5ac5e674a2e49914be9f49309e6d77454003a39eb2f6
     or http://127.0.0.1:8888/?token=657a5ac5e674a2e49914be9f49309e6d77454003a39eb2f6

```

## 3) FAQ

This section of the repo will be periodically updated to represent common questions which may arise around its use. If you detect any problems/bugs, please [create an issue](https://github.com/1272371/DN3_Classifiers/discussions) and we will do our best to resolve it as quickly as possible.

We wish you all the best in your learning experience :rocket:

![Explore Data Science Academy](resources/imgs/EDSA_logo.png)
