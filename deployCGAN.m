% JN Kather, J Krause 2019-2020
% This is part of the deepGAN repository
% License: see separate LICENSE file 
% 
% documentation for this function:
% this will deploy a trained GAN and use it to generate synthetic images

function deployCGAN(varargin)

addpath(genpath('./subGan'));
disp('deploy CGAN version 2')
rng('shuffle');
%% parse input arguments, for documentation see inputParserCGAN()
iPrs = inputParserCGAN(varargin);  % get input parser, define default values
cnst.expID = ['deploy_',randseq(10,'alphabet','AA')];
cnst = copyfields(cnst,iPrs.Results,fieldnames(iPrs.Results)); % apply input
cnst %#ok
sq = @(varargin) varargin;

%% load and prepare data
loadModel = fullfile(cnst.masterOutputFolder,cnst.relPathTrainedModel);
disp(['-- starting to load ',loadModel]);
tNet = load(loadModel,'cnst','dlnetGenerator','dlnetDiscriminator','dlZValidation','dlTValidation');
disp(['-- successfully loaded model ',tNet.cnst.expID, ' in deploy run ',cnst.expID]);

%% generate images

numLatent = max(tNet.dlnetGenerator.Layers(1).InputSize);
numClass  = tNet.cnst.numClass; 

% prepare arrays
rng('default'); % reproducibility

numMb = ceil(cnst.numGenerate/cnst.miniBatchSize); % number of mini batches

for mb = 1:numMb
disp(['--- starting mini batch ',num2str(mb),' of ',num2str(numMb)]);
%Z = randn(1,1,numLatent,cnst.miniBatchSize*numClass,'single');
Z = randn(1,1,numLatent,cnst.miniBatchSize*numClass,'single');
T = single(repmat(1:numClass,[1 cnst.miniBatchSize]));
T = permute(T,[1 3 4 2]);
dlZ = dlarray(Z, 'SSCB');
dlT = dlarray(T, 'SSCB');

% use GPU if possible
if canUseGPU
    dlZ = gpuArray(dlZ);
    dlT = gpuArray(dlT);
else
    warning('non-GPU mode');
end

dlXGeneratedValidation = predict(tNet.dlnetGenerator,dlZ,dlT);
disp('----- generated images');

if cnst.getScore
    % get discriminator-predicted realism score
    dlYPredGenerated = forward(tNet.dlnetDiscriminator, dlXGeneratedValidation, dlT);
    probGenerated = squeeze(extractdata(sigmoid(dlYPredGenerated)));
    disp('----- extracted discriminator sccores');
else
    probGenerated = nan(numel(dlT),1);
end

% extract generated images
I = rescale(extractdata(dlXGeneratedValidation));
labels = squeeze(cell2mat(sq(T)));

% prepare save folders
if cnst.doSave
    for ic = 1:numClass
        outFolder{ic} = fullfile(cnst.masterOutputFolder,tNet.cnst.expID,['gen_',cnst.expID],num2str(ic));
        mkdir(outFolder{ic});
    end
end
        
%iterate images
disp(['----- iterating images - ',num2str(size(I,4))]);
for numI = 1:size(I,4)
    currImage = squeeze(I(:,:,:,numI));
    currLabel = num2str(labels(numI));
    currScore = probGenerated(numI);
    
    if cnst.doPlot
        imshow(currImage)
        title(currLabel);
        drawnow
        pause
    end
    
    if cnst.doSave
        imwrite(currImage,fullfile(outFolder{labels(numI)},...
            ['gen','_score_',num2str(currScore,4),'_cl_',currLabel,'_mb_',num2str(mb,'%04.0f'),'_img_',num2str(numI,'%08.0f'),'.jpg']));
    end
end
end
end
