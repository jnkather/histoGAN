% provided by Mathworks, re-used for this project


function [lossGenerator, lossDiscriminator] = ganLoss(scoresReal,scoresGenerated,labelSmoothing)

% Calculate losses for the discriminator network.
lossGenerated = -mean(log(1 - scoresGenerated));
%lossReal = -mean(log(labelSmoothing*sigmoid(scoresReal)));
lossReal = -mean(log(labelSmoothing*scoresReal));

% Combine the losses for the discriminator network.
lossDiscriminator = lossReal + lossGenerated;

% Calculate the loss for the generator network.
lossGenerator = -mean(log(scoresGenerated));

end