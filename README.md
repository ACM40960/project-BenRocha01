# A Chess Engine Framework for Neural Network Evaluation functions
Made by:
Bernardo Rocha 23214074
requires:
Python 3.8
TensorFlow 2.13
Numpy 1.24.3
chess 1.10

## Basic Overview
A code framework to program chess engines in Python.
This framework was then used to created Neural Network based evaluation functions
It has searching, evaluation and feature functions, that I explain better later

## Table of contents
<ol>
    
<li>Instalation</li>
<li>Usage</li>
<li>Structure of the code file</li>

</ol>

## Instalation
To use this project, download the Project_code.py file
Then call it from a python interface as you would a normal package

## Usage
To create a chess engine (here called BotPlayer) you need 2 things.

### 1 create a Searcher using SearcherDirector
SearcherDirector has the written searching functions as functions for the class.
Currently there are 5 functions: minimax, Minimax_NN, AB_pruning, AB_prunning_NN and ID_AB_pruning
All of them have a parameter that represents depth.

### 2 create a list of Feature using FeatureDirector (Only for hand-crafted evaluation)
Before starting this step you need to know if you want a NN evaluation function or a Manual evaluatin function.
The FeatureDirector has all the functions already prepared, you just need to set a a parameter.

### 3 create an Evaluator using the EvaluatorDirector
This will take the list of features as argument if you are creating a hand-crafted function
If creating a NN evaluation function the arguments are the model and the minply

There are 9 models:
"Single256","Single128","Single64","Single32","Pair256","Double128","Single256_Double256_128","Single128_Double128"
and 5 possible values for minply
"0","15","30","45","60"

### 4 Assemble everything under the BotPlayer
Create an object class BotPlayer that takes 3 arguments, a name (string, not relevant), a Searcher and an Evaluator

### 5 play the simulations
Use the function simulateChessGame or simulateMultipleGames, using 2 Players of your choice
(They can be the same, a HumanPlayer works aswell and needs no arguments when being created)







## Structure of the code file

<ol>
<li>Imports</li>
<li>Global Variable</li>
<li>Main functions</li>
<li>Playing functions</li>
<li>Player</li>
<li>Search</li>
<li>Evaluation
    <ol>
    <li>Manual Evaluation<ol>
        <li>Manual calc_funct</li>
    </ol></li>
    <li>NNEvaluation<ol>
        <li>NeuralNetworks</li>
    </ol></li>

</ol></li>
<li>Trait<ol>
    <li>ManualTrait</li>
    <li>NNTrait</li>
</ol> </li>
<li>Builders</li>
<li>Misc functions</li>
<li>Preset instances<ol>
    <li>Lists of options</li>
    <li>Traits</li>
    <li>Evaluations</li>
    <li>Bots</li>
</ol></li>

</ol>
