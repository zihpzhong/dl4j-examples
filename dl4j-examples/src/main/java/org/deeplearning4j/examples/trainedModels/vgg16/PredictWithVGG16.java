package org.deeplearning4j.examples.trainedModels.vgg16;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by susaneraly on 1/12/17.
 *
 * This example demonstrates how to import Keras Models into dl4j given the weights and config.
 * //FIX ME - citation? mention Keras?
 * Here we will
 * will glob for images in a given directory and report
 * classifications after running them through VGG16
 *
 * It will check for keras
 */
public class PredictWithVGG16 {

    protected static Logger logger= LoggerFactory.getLogger(PredictWithVGG16.class);
    public static final String IMG_DIR = "DIR_WITH_YOUR_IMAGES";
    public static final int batchSize = 10;
    public static final boolean recursiveFind = true;

    public static void main(String[] args) {

        //ComputationGraph vggNet= KerasModelImport.importKerasModelAndWeights(MODEL_DIR + "/vgg16.json", MODEL_DIR + "/vgg16.h5");
        logger.info("Loading VGG..");
        ComputationGraph vggNet = KerasTrainedModels.VGG16Import();

        //??Should be a generic iterator?
        VGG16ImageIterator iterator = KerasTrainedModels.VGG16ImageIterator(IMG_DIR,batchSize,recursiveFind);

        while (iterator.hasNext()) {
            INDArray preProcessedImages = iterator.next();
            String [] imageFileNames = iterator.fileNames();
            INDArray[] outputArray = vggNet.output(false,preProcessedImages);
            INDArray output = Nd4j.concat(0,outputArray);
            INDArray predictedClass = Nd4j.argMax(output,1);
            String [] predictedClassNames = KerasTrainedModels.VGG16ClassNames(predictedClass);
        }

    }


}
