% JN Kather, J Krause 2019-2020
% This is part of the deepGAN repository
% License: see separate LICENSE file 
% 
% documentation for this function:
% this is a the input parser for the GAN project

function p=inputParserCGAN(myVargs)

    p=inputParser;            % prepare to parse user input
    p.CaseSensitive=false;    % input is not case sensitive
    p.PartialMatching=false;  % strict input parser options
    p.KeepUnmatched=false;    % strict input parser options

    %% general options for train and deploy
    addParameter(p,'gpuDev',1,@isnumeric);
    addParameter(p,'masterOutputFolder',fullfile('D:/GANchkpts/'),@ischar); % output folder
    addParameter(p,'miniBatchSize',512,@isnumeric);  % default 128
        
    %% for training GANs
    % general options for training
    addParameter(p,'inputDir',fullfile('D:\TCGA-CRC-DX\EXPORT_VPGCCKTHIFDK-normalized\TRAIN'),@ischar);
    addParameter(p,'doFlip',true,@islogical); % random xy flips for input imgs
    addParameter(p,'lives',5,@isnumeric); % how many times can it fall back to checkpoint
    addParameter(p,'alternateNet',1,@isnumeric); % use alternate architecture, default 1
    
    % generator options
    addParameter(p,'inPx',64,@isnumeric); % image input pixels, default 64
    addParameter(p,'numLatent',100,@isnumeric); % latent input dimension
    addParameter(p,'embeddingDim',50,@isnumeric); % embedding dimension
    addParameter(p,'numFilt',64,@isnumeric); % number of filters
    addParameter(p,'filtSize',5,@isnumeric);  % filter size
    addParameter(p,'projSize',[4 4 1024],@isnumeric); % projection size
    addParameter(p,'generatorBnEps',5e-5,@isnumeric); % default 1e-5, batch normalization epsilon for generator
    addParameter(p,'discriminatorBnEps',1e-5,@isnumeric); % default 1e-5, batch normalization epsilon for discriminator

    % discriminator options
    addParameter(p,'dropout',0.25,@isnumeric);
    addParameter(p,'scaleRelu',0.2,@isnumeric);

    % training options
    addParameter(p,'numEpochs',500,@isnumeric);
    addParameter(p,'learnRate',0.0002,@isnumeric);
    addParameter(p,'gradientDecay',0.5,@isnumeric);
    addParameter(p,'squaredGradientDecay',0.999,@isnumeric);
    addParameter(p,'execEnvironment','auto',@ischar);
    addParameter(p,'valFrequency',10,@isnumeric);  % validation frequency
    addParameter(p,'valChkpt',2000,@isnumeric);    % save checkpoint frequency
    addParameter(p,'flipFactor',0.5,@isnumeric);   % flip labels to handicap discriminator
    addParameter(p,'numValidationImgs',5,@isnumeric); % how many validation images to plot
    addParameter(p,'labelSmoothing',0.9,@isnumeric); % default 1, label smoothing in loss function

    %% for deployment of GANS
    addParameter(p,'relPathTrainedModel','',@ischar); % path to trained model, dependent on the master path
    addParameter(p,'numGenerate',10,@isnumeric); % how many images to generate in total
    addParameter(p,'doSave',true,@islogical); % save resulting images
    addParameter(p,'doPlot',false,@islogical); % show resulting images
    addParameter(p,'getScore',false,@islogical); % calculate score for each image
    
    parse(p,myVargs{:});        % parse input arguments
    
end
    
    