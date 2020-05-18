% JN Kather, J Krause 2019-2020
% This is part of the deepGAN repository
% License: see separate LICENSE file 
% 
% documentation for this function:
% this can be used to train a GAN on a fixed set of images

function runCGAN(varargin)

addpath(genpath('./subGan'));

%% parse input arguments, for documentation see inputParserCGAN()
iPrs = inputParserCGAN(varargin);  % get input parser, define default values
cnst.expID = randseq(10,'alphabet','AA');
cnst = copyfields(cnst,iPrs.Results,fieldnames(iPrs.Results)); % apply input
cnst %#ok

%% load and prepare data
datasetFolder = fullfile(cnst.inputDir);
imds = imageDatastore(datasetFolder, 'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

if cnst.doFlip
    augm = imageDataAugmenter('RandXReflection',true,'RandYReflection',true);
    disp('- will xy flip images');
else
    augm = imageDataAugmenter();
    disp('- will *not* xy flip images');
end

augimds = augmentedImageDatastore([cnst.inPx,cnst.inPx],imds,'DataAugmentation',augm);

%% apply settings
augimds.MiniBatchSize = cnst.miniBatchSize;
cnst.inputSize = [cnst.inPx,cnst.inPx,3];
cnst.numClass = numel(categories(imds.Labels));
cnst.outputFolder = fullfile(cnst.masterOutputFolder,cnst.expID);
mkdir(cnst.outputFolder);
gpuDevice(cnst.gpuDev);

%% prepare networks
dlnetGenerator = getGeneratorNetwork(cnst);
dlnetDiscriminator = getDiscriminatorNetwork(cnst);

%% train GAN

% preallocate
trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];
lives = cnst.lives;

% visualize
f = figure;
clf
f.Position(3) = 2*f.Position(3);
imageAxes = subplot(1,2,1);
sax = subplot(1,2,2);
lineGen = animatedline(sax,'Color',[0 0.447 0.741],'LineWidth',1.2);
lineDis = animatedline(sax, 'Color', [0.85 0.325 0.098],'LineWidth',1.2);
legend('Generator','Discriminator');
ylim([0 1])
xlabel("Iteration")
ylabel("Score")
grid on

% prepare arrays
ZValidation = randn(1,1,cnst.numLatent,cnst.numValidationImgs*cnst.numClass,'single');
TValidation = single(repmat(1:cnst.numClass,[1 cnst.numValidationImgs]));
TValidation = permute(TValidation,[1 3 4 2]);
dlZValidation = dlarray(ZValidation, 'SSCB');
dlTValidation = dlarray(TValidation, 'SSCB');

% use GPU if possible
if (cnst.execEnvironment == "auto" && canUseGPU) || cnst.execEnvironment == "gpu"
    dlZValidation = gpuArray(dlZValidation);
    dlTValidation = gpuArray(dlTValidation);
end

iteration = 0;
start = tic;

% iterate epochs
for epoch = 1:cnst.numEpochs
    
    % Reset and shuffle datastore.
    reset(augimds);
    augimds = shuffle(augimds);
    
    % Loop over mini-batches.
    while hasdata(augimds)
        
        % Read mini-batch of data and generate latent inputs for the
        % generator network.
        data = read(augimds);
        
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < cnst.miniBatchSize
            % iteration does not take place
            continue
        else
            % iteration takes place
            iteration = iteration + 1;
        end
        
        X = cat(4,data{:,1}{:});
        X = single(X);
        
        T = single(data.response);
        T = permute(T,[2 3 4 1]);
        
        Z = randn(1,1,cnst.numLatent,cnst.miniBatchSize,'single');
        
        % rescale the images in the range [-1 1].
        X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
        
        % Convert mini-batch of data to dlarray and specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');
        dlT = dlarray(T, 'SSCB');
        
        % If training on a GPU, convert data to gpuArray.
        if (cnst.execEnvironment == "auto" && canUseGPU) || cnst.execEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlZ = gpuArray(dlZ);
            dlT = gpuArray(dlT);
        end
        
        % Evaluate the model gradients and the generator state
        [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, ...
            scoreDiscriminator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, ...
                dlX, dlT, dlZ, cnst.flipFactor, cnst.labelSmoothing);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            cnst.learnRate, cnst.gradientDecay, cnst.squaredGradientDecay);
        
        % Update the generator network parameters.
        [dlnetGenerator,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            cnst.learnRate, cnst.gradientDecay, cnst.squaredGradientDecay);
        
        % Every cnst.valFrequency iterations, display batch of generated images using the
        % held-out generator input.
        if mod(iteration,cnst.valFrequency) == 0 || iteration == 1 || mod(iteration,cnst.valChkpt) == 0 
            
            try 
                % Generate images using the held-out generator input.
                dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation,dlTValidation);
            catch myError
                if lives > 0 && iteration > 1
                    
                    warning(['-- generation of showcase images failed in iteration ',num2str(iteration)]);
                    disp(['---- falling back to last checkpoint... remaining lives: ',num2str(lives)]);
                    lives = lives - 1;
                    load(fullfile(cnst.outputFolder,chkptfile),'dlnetGenerator','iteration');
                    disp(['---- loaded last checkpoint, new iteration: ',num2str(iteration)]);
                    
                    continue
                else
                    disp('--- all lives are gone.. will abort');
                    rethrow(myError);
                end
            end
                
            % Tile and rescale the images in the range [0 1].
            I = imtile(extractdata(dlXGeneratedValidation), ...
                'GridSize',[cnst.numValidationImgs cnst.numClass]);
            I = rescale(I);
            
            % Display the images.
            subplot(1,2,1);
            image(imageAxes,I)
            xticklabels([]);
            yticklabels([]);
            axis equal tight off
            title("Generated Images");
        end
        
        % Every cnst.valChkpt iterations, save a model checkpoint
        if mod(iteration,cnst.valChkpt) == 0 || iteration == 1 
            
            % save the images
            imwrite(I,fullfile(cnst.outputFolder,['chkpt_',num2str(iteration,'%08.0f'),'.png']));
            
            % save the models
            [xgen,ygen] = getpoints(lineGen);
            [xdis,ydis] = getpoints(lineDis);
            chkptfile = ['chkpt_',num2str(iteration,'%08.0f'),'.mat'];
            save(fullfile(cnst.outputFolder,chkptfile),...
                'cnst','dlnetGenerator','dlnetDiscriminator',...
                'dlZValidation','dlTValidation',...
                'iteration','epoch','xgen','ygen','xdis','ydis');
            
            disp(['--- saved checkpoint at iteration ',num2str(iteration,'%08.0f')]);
            
        end
        
        % Update the scores plot every iteration
        subplot(1,2,2)
        addpoints(lineGen,iteration,...
            double(gather(extractdata(scoreGenerator))));
        addpoints(lineDis,iteration,...
            double(gather(extractdata(scoreDiscriminator))));
        
        % Update the title with training progress information.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title("Epoch: " + epoch + ", " + "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(D))
        
        drawnow
    end
end

end
