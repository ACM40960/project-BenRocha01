Just a temporary placeholder

The main code right now is Project_code. The rest are mostly remainders of previous versions and I will discard them later. 

Bernardo Rocha 23214074

decided to only use sequential because going to higher complexity models would 
lead down a rabbit hoe that doesn't fall into the perview of this module

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