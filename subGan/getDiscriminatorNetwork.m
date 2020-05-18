% provided by Mathworks, re-used for this project

function dlnetDiscriminator = getDiscriminatorNetwork(cnst)

switch cnst.inPx
    case 64
    layersDiscriminator = [
        imageInputLayer(cnst.inputSize,'Normalization','none','Name','images')
        dropoutLayer(cnst.dropout,'Name','dropout')
        concatenationLayer(3,2,'Name','cat')
            convolution2dLayer(cnst.filtSize,cnst.numFilt,'Stride',2,'Padding','same','Name','conv1')
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu1')
            convolution2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Padding','same','Name','conv2')
        batchNormalizationLayer('Name','bn2','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu2')
            convolution2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Padding','same','Name','conv3')
        batchNormalizationLayer('Name','bn3','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu3')
            convolution2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4')
        batchNormalizationLayer('Name','bn4','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4')
        convolution2dLayer(4,1,'Name','conv5')];
    case 128
    layersDiscriminator = [
        imageInputLayer(cnst.inputSize,'Normalization','none','Name','images')
        dropoutLayer(cnst.dropout,'Name','dropout')
        concatenationLayer(3,2,'Name','cat')
            convolution2dLayer(cnst.filtSize,cnst.numFilt,'Stride',2,'Padding','same','Name','conv1')
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu1')
            convolution2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Padding','same','Name','conv2')
        batchNormalizationLayer('Name','bn2','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu2')
            convolution2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Padding','same','Name','conv3')
        batchNormalizationLayer('Name','bn3','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu3')
            convolution2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4')
        batchNormalizationLayer('Name','bn4','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4')
            convolution2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4b')
        batchNormalizationLayer('Name','bn4b','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4b')
        convolution2dLayer(4,1,'Name','conv5')];
    case 256
    layersDiscriminator = [
        imageInputLayer(cnst.inputSize,'Normalization','none','Name','images')
        dropoutLayer(cnst.dropout,'Name','dropout')
        concatenationLayer(3,2,'Name','cat')
            convolution2dLayer(cnst.filtSize,cnst.numFilt,'Stride',2,'Padding','same','Name','conv1')
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu1')
            convolution2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Padding','same','Name','conv2')
        batchNormalizationLayer('Name','bn2','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu2')
            convolution2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Padding','same','Name','conv3')
        batchNormalizationLayer('Name','bn3','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu3')
            convolution2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4')
        batchNormalizationLayer('Name','bn4','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4')
            convolution2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4b')
        batchNormalizationLayer('Name','bn4b','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4b')
            convolution2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4c')
        batchNormalizationLayer('Name','bn4c','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4c')
        convolution2dLayer(4,1,'Name','conv5')];
    case 512
         if false && isfield(cnst,'alternateNet') && (cnst.alternateNet == 3)
    layersDiscriminator = [
        imageInputLayer(cnst.inputSize,'Normalization','none','Name','images')
        dropoutLayer(cnst.dropout,'Name','dropout')
        concatenationLayer(3,2,'Name','cat')
            convolution2dLayer(cnst.filtSize,cnst.numFilt,'Stride',2,'Padding','same','Name','conv1')
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu1')
            convolution2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Padding','same','Name','conv2')
        batchNormalizationLayer('Name','bn2','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu2')
            convolution2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Padding','same','Name','conv3')
        batchNormalizationLayer('Name','bn3','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu3')
            convolution2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4')
        batchNormalizationLayer('Name','bn4','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4')
            convolution2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4b')
        batchNormalizationLayer('Name','bn4b','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4b')
            convolution2dLayer(cnst.filtSize,10*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4c')
        batchNormalizationLayer('Name','bn4c','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4c')
            convolution2dLayer(cnst.filtSize,12*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4d')
        batchNormalizationLayer('Name','bn4d','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4d')
        convolution2dLayer(4,1,'Name','conv5')]; 
         else
    layersDiscriminator = [
        imageInputLayer(cnst.inputSize,'Normalization','none','Name','images')
        dropoutLayer(cnst.dropout,'Name','dropout')
        concatenationLayer(3,2,'Name','cat')
            convolution2dLayer(cnst.filtSize,cnst.numFilt,'Stride',2,'Padding','same','Name','conv1')
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu1')
            convolution2dLayer(cnst.filtSize,2*cnst.numFilt,'Stride',2,'Padding','same','Name','conv2')
        batchNormalizationLayer('Name','bn2','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu2')
            convolution2dLayer(cnst.filtSize,4*cnst.numFilt,'Stride',2,'Padding','same','Name','conv3')
        batchNormalizationLayer('Name','bn3','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu3')
            convolution2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4')
        batchNormalizationLayer('Name','bn4','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4')
            convolution2dLayer(cnst.filtSize,6*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4b')
        batchNormalizationLayer('Name','bn4b','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4b')
            convolution2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4c')
        batchNormalizationLayer('Name','bn4c','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4c')
            convolution2dLayer(cnst.filtSize,8*cnst.numFilt,'Stride',2,'Padding','same','Name','conv4d')
        batchNormalizationLayer('Name','bn4d','epsilon',cnst.discriminatorBnEps)
        leakyReluLayer(cnst.scaleRelu,'Name','lrelu4d')
        convolution2dLayer(4,1,'Name','conv5')];
         end
    otherwise
        error('not yet implemented');
end

lgraphDiscriminator = layerGraph(layersDiscriminator);
layers = [
    imageInputLayer([1 1],'Name','labels','Normalization','none')
    embedAndReshapeLayer(cnst.inputSize,cnst.embeddingDim,cnst.numClass,'emb')];
lgraphDiscriminator = addLayers(lgraphDiscriminator,layers);
lgraphDiscriminator = connectLayers(lgraphDiscriminator,'emb','cat/in2');
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);

end