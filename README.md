# Basic Overview
A code framework to program chess engines in Python.
This framework was then used to created Neural Network based evaluation functions








# Structure of the framework

The base function of the framework is simulateChessGame. This function takes 2 arguments, both are Players, a costum objects.

The structure of a BotPlayer is the following:


During operation the Searcher explores the game tree, using the Evaluator to evaluate and choose from the explored leafs.

There are 2 main types of Evaluators, that have the same structure but operate differently. The Manual Evaluators and the NN Evaluators.
An Evaluator is composed of a list of Traits and a model. The Traits get information from the board and the model calculates the board value based on that information.






# Extra
Just a temporary placeholder

The main code right now is Project_code. The rest are mostly remainders of previous versions and I will discard them later. 

Bernardo Rocha 23214074

decided to only use sequential because going to higher complexity models would 
lead down a rabbit hoe that doesn't fall into the perview of this module


During the project I had to deal with the problem of calculations on the GPU or CPU. There are different approaches for both, but at the start I'll stick with tensorflow and GPU based models.

# How to generate a BotPlayer
<ol>
<li>Decide if you want a Manual or NN bot</li>
<li>Get a list of Traits accordingly to your choice</li>
<li><ul>
<li>If you chose an NN bot choose a NN model as calc_funct</li>
<li>If you chose a Manual bot choose a manual calc_funct</li>
</ul></li>
<li>Create an Evaluator using the list of Traits and the calc_funct</li>
<li>Create a Searcher using a searching function and a parameter</li>
<li>Create a BotPlayer using the Evaluator and the Searcher defined previously</li>
<li>You are ready use it to play or training</li>
</ol>




# Structure of the code file

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