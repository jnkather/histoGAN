% provided by Mathworks, re-used for this project

function dlnetGenerator = getGeneratorNetwork(cnst)

switch cnst.inPx
    case 64
    layersGenerator = [
        imageInputLayer([1 1 cnst.numLatent],'Normalization','none','Name','noise')
        projectAndReshapeLayer(cnst.projSize,cnst.numLatent,'proj');
        concatenationLayer(3,2,'Name','cat');
            transposedConv2dLayer(cnst.filtSize,4*cnst.numFilt,'Name','tconv1')
        batchNormalizationLayer('Name','bn1','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu1')
            transposedConv2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv2')
        batchNormalizationLayer('Name','bn2','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu2')
            transposedConv2dLayer(cnst.filtSize,1*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv3')
        batchNormalizationLayer('Name','bn3','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu3')
            transposedConv2dLayer(cnst.filtSize,3,'Stride',2,'Cropping','same','Name','tconv4')
        tanhLayer('Name','tanh')];
    case 128
    layersGenerator = [
        imageInputLayer([1 1 cnst.numLatent],'Normalization','none','Name','noise')
        projectAndReshapeLayer(cnst.projSize,cnst.numLatent,'proj');
        concatenationLayer(3,2,'Name','cat');
            transposedConv2dLayer(cnst.filtSize,8*cnst.numFilt,'Name','tconv1b')
        batchNormalizationLayer('Name','bn1b','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu1b')
            transposedConv2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1')
        batchNormalizationLayer('Name','bn1','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu1')
            transposedConv2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv2')
        batchNormalizationLayer('Name','bn2','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu2')
            transposedConv2dLayer(cnst.filtSize,1*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv3')
        batchNormalizationLayer('Name','bn3','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu3')
            transposedConv2dLayer(cnst.filtSize,3,'Stride',2,'Cropping','same','Name','tconv4')
        tanhLayer('Name','tanh')];
    case 256
    layersGenerator = [
        imageInputLayer([1 1 cnst.numLatent],'Normalization','none','Name','noise')
        projectAndReshapeLayer(cnst.projSize,cnst.numLatent,'proj');
        concatenationLayer(3,2,'Name','cat');
            transposedConv2dLayer(cnst.filtSize,8*cnst.numFilt,'Name','tconv1a')
        batchNormalizationLayer('Name','bn1a','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu1a')
            transposedConv2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1b')
        batchNormalizationLayer('Name','bn1b','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu1b')
            transposedConv2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1')
        batchNormalizationLayer('Name','bn1','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu1')
            transposedConv2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv2')
        batchNormalizationLayer('Name','bn2','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu2')
            transposedConv2dLayer(cnst.filtSize,1*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv3')
        batchNormalizationLayer('Name','bn3','epsilon',cnst.generatorBnEps)
        reluLayer('Name','relu3')
            transposedConv2dLayer(cnst.filtSize,3,'Stride',2,'Cropping','same','Name','tconv4')
        tanhLayer('Name','tanh')];
    case 512
        if isfield(cnst,'alternateNet') && (cnst.alternateNet == 3)
        layersGenerator = [
            imageInputLayer([1 1 cnst.numLatent],'Normalization','none','Name','noise')
            projectAndReshapeLayer(cnst.projSize,cnst.numLatent,'proj');
            concatenationLayer(3,2,'Name','cat');
                transposedConv2dLayer(cnst.filtSize,12*cnst.numFilt,'Name','tconv0')
            batchNormalizationLayer('Name','bn0','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu0')
                transposedConv2dLayer(cnst.filtSize,10*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1a')
            batchNormalizationLayer('Name','bn1a','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu1a')
                transposedConv2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1b')
            batchNormalizationLayer('Name','bn1b','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu1b')
                transposedConv2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1')
            batchNormalizationLayer('Name','bn1','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu1')
                transposedConv2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv2')
            batchNormalizationLayer('Name','bn2','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu2')
                transposedConv2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv3')
            batchNormalizationLayer('Name','bn3','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu3')
                transposedConv2dLayer(cnst.filtSize,3,'Stride',2,'Cropping','same','Name','tconv4')
            tanhLayer('Name','tanh')]; 
        elseif isfield(cnst,'alternateNet') && (cnst.alternateNet == 2)
            % alternate 512 net
        layersGenerator = [
            imageInputLayer([1 1 cnst.numLatent],'Normalization','none','Name','noise')
            projectAndReshapeLayer(cnst.projSize,cnst.numLatent,'proj');
            concatenationLayer(3,2,'Name','cat');
                transposedConv2dLayer(cnst.filtSize,8*cnst.numFilt,'Name','tconv0')
            batchNormalizationLayer('Name','bn0','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu0')
                transposedConv2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1a')
            batchNormalizationLayer('Name','bn1a','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu1a')
                transposedConv2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1b')
            batchNormalizationLayer('Name','bn1b','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu1b')
                transposedConv2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1')
            batchNormalizationLayer('Name','bn1','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu1')
                transposedConv2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv2')
            batchNormalizationLayer('Name','bn2','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu2')
                transposedConv2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv3')
            batchNormalizationLayer('Name','bn3','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu3')
                transposedConv2dLayer(cnst.filtSize,3,'Stride',2,'Cropping','same','Name','tconv4')
            tanhLayer('Name','tanh')]; 
        else
            % default 512 net
        layersGenerator = [
            imageInputLayer([1 1 cnst.numLatent],'Normalization','none','Name','noise')
            projectAndReshapeLayer(cnst.projSize,cnst.numLatent,'proj');
            concatenationLayer(3,2,'Name','cat');
                transposedConv2dLayer(cnst.filtSize,8*cnst.numFilt,'Name','tconv0')
            batchNormalizationLayer('Name','bn0','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu0')
                transposedConv2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1a')
            batchNormalizationLayer('Name','bn1a','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu1a')
                transposedConv2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1b')
            batchNormalizationLayer('Name','bn1b','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu1b')
                transposedConv2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv1')
            batchNormalizationLayer('Name','bn1','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu1')
                transposedConv2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv2')
            batchNormalizationLayer('Name','bn2','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu2')
                transposedConv2dLayer(cnst.filtSize,1*cnst.numFilt,'Stride',2,'Cropping','same','Name','tconv3')
            batchNormalizationLayer('Name','bn3','epsilon',cnst.generatorBnEps)
            reluLayer('Name','relu3')
                transposedConv2dLayer(cnst.filtSize,3,'Stride',2,'Cropping','same','Name','tconv4')
            tanhLayer('Name','tanh')];
        end
    otherwise
        error('not yet implemented');
end
    
lgraphGenerator = layerGraph(layersGenerator);
layers = [
    imageInputLayer([1 1],'Name','labels','Normalization','none')
    embedAndReshapeLayer(cnst.projSize(1:2),cnst.embeddingDim,cnst.numClass,'emb')];
lgraphGenerator = addLayers(lgraphGenerator,layers);
lgraphGenerator = connectLayers(lgraphGenerator,'emb','cat/in2');
dlnetGenerator = dlnetwork(lgraphGenerator);

end