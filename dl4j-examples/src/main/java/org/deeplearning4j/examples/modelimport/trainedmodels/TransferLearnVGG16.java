package org.deeplearning4j.examples.modelimport.trainedmodels;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Please run "FeaturizeVGG16NoTop" first
 *  where images in the animals dataset are featurized with vgg16-notop (imported from keras configs) and saved to separate train and test folders
 *
 * Here we train a simple MLP classifier on this data to demonstrate transfer learning with a previously trained model
 * NOTE: This is an example that runs on a toy dataset of 80 images written with the intent of demonstrating the concepts.
 *
 * //FIXME
 * Citation
 */
public class TransferLearnVGG16 {

    private static final Logger log = LoggerFactory.getLogger(TransferLearnVGG16.class);
    protected static int[] inputShape = TrainedModels.VGG16NOTOP.getOuputShape();
    protected static int channels = inputShape[1];
    protected static int height = inputShape[2];
    protected static int width = inputShape[3];
    protected static int nEpochs = 10;

    public static void main(String [] args) {

        log.info("Loading presaved featurized image datasets....");
        DataSetIterator existingTrainingData = new ExistingMiniBatchDataSetIterator(new File("trainFolder"),"featurized-%d.bin");
        DataSetIterator trainIter= new AsyncDataSetIterator(existingTrainingData);
        NormalizerStandardize scaler = new NormalizerStandardize();
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        DataSetIterator existingTestData = new ExistingMiniBatchDataSetIterator(new File("testFolder"),"featurized-%d.bin");
        DataSetIterator testIter = new AsyncDataSetIterator(existingTestData);
        testIter.setPreProcessor(scaler);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(124)
            .iterations(1)
            .activation(Activation.RELU)
            .learningRate(1e-4)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .list()
            .layer(0, new DenseLayer.Builder()
                    .nIn(height*width*channels)
                    .nOut(256).dropOut(0.25).build())
            /*.layer(1, new DenseLayer.Builder()
                    .nOut(128).dropOut(0.5).build())
                    */
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                    .nOut(FeaturizeVGG16NoTop.numLabels)
                    .activation(Activation.SOFTMAX)
                    .build())
            .pretrain(false)
            .backprop(true)
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(trainIter);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(FeaturizeVGG16NoTop.numLabels);
            while(testIter.hasNext()){
                DataSet ds = testIter.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);

            }
            log.info(eval.stats());
            testIter.reset();
        }

    }


}
