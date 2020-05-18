% JN Kather, J Krause 2019-2020
% This is part of the deepGAN repository
% License: see separate LICENSE file 
% 
% documentation for this function:
% this is a debug function which will help you to find out if the generator
% and discriminator are compatible

function debugCGAN(varargin)

addpath(genpath('./subGan'));

iPrs = inputParserCGAN(varargin);  % get input parser, define default values
cnst.expID = randseq(10,'alphabet','AA');
cnst = copyfields(cnst,iPrs.Results,fieldnames(iPrs.Results)); % apply input
cnst.inputSize = [cnst.inPx,cnst.inPx,3];
cnst.numClass = 2;
cnst %#ok

% prepare data
ZValidation = randn(1,1,cnst.numLatent,cnst.numValidationImgs*cnst.numClass,'single');
TValidation = single(repmat(1:cnst.numClass,[1 cnst.numValidationImgs]));
TValidation = permute(TValidation,[1 3 4 2]);
dlZValidation = dlarray(ZValidation, 'SSCB');
dlTValidation = dlarray(TValidation, 'SSCB');

disp('simulating generator network');
dlnetGenerator = getGeneratorNetwork(cnst);
disp(['--- input: ',num2str(size(ZValidation))]);
dly = predict(dlnetGenerator,dlZValidation,dlTValidation); 
disp(['--- output: ',num2str(size(extractdata(dly)))]);

disp('simulating discriminator network');
dlnetDiscriminator = getDiscriminatorNetwork(cnst);
disp('-- discriminator');
disp(['--- input: ',num2str(size(extractdata(dly)))]);
dlz = predict(dlnetDiscriminator,dly,dlTValidation); 
disp(['--- output: ',num2str(size(extractdata(dlz)))]);

end
