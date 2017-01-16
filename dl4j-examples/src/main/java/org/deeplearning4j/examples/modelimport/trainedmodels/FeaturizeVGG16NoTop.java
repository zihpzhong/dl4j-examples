package org.deeplearning4j.examples.modelimport.trainedmodels;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 * Illustrates using VGG16-noTop (VGG16 without the last three fully connected layers) to "featurize" an image dataset.
 * VGG16-noTop is loaded into DL4J via Keras weights and json configs. The required H5 and json configs will be downloaded to ~/.dl4j/trainedmodels/vgg16notop during the first run.
 * Featurized trained and test data are stored for use later. The class "TransferLearnVGG16" trains a small MLP classifier on the saved data.
 *
 * Uses the image pipeline and dataset from the dl4j AnimalsClassification example.
 * Data set in dl4j-examples/src/main/resources/animals with images of bear, duck, deer and turtle.
 * credits: nyghtowl
 *
 * Image References:
 *  - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
 *  - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/leonyao_final.pdf
 *
 * //FIXME - Keras? VGG paper? ImageNet?
 * Citation:
 *
 * @author susaneraly
 */
public class FeaturizeVGG16NoTop {

    protected static final Logger log = LoggerFactory.getLogger(FeaturizeVGG16NoTop.class);
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int numExamples = 80;
    protected static int numLabels = 4;
    protected static int batchSize = 16;
    protected static double splitTrainTest = 0.75;
    protected static ParentPathLabelGenerator labelMaker;

    protected static final int height = 224;
    protected static final int width = 224;
    protected static int channels = 3;

    protected static ComputationGraph vgg16NoTop;

    public static void main(String [] args) throws Exception {

        //Load "VGG16,no top" into dl4j using KerasModelImporter
        TrainedModelHelper helper = new TrainedModelHelper(TrainedModels.VGG16NOTOP);
        //NOTE: Once I upload these files these methods go away and will get downloaded to the user's home dir during the first run...
        helper.setPathToH5("/Users/susaneraly/SKYMIND/kerasImport/VGG16/saved/vgg16notop.h5");
        helper.setPathToJSON("/Users/susaneraly/SKYMIND/kerasImport/VGG16/saved/vgg16notop.json");
        vgg16NoTop = helper.loadModel();

        log.info("Setting up the image pipeline and conducting the train and test splits....");
        //Refer to the AnimalsClassification example for more details
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals/");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, numExamples * splitTrainTest, numExamples * (1 - splitTrainTest));
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        log.info("Running through the train split...");
        File trainFolder = new File("trainFolder");
        trainFolder.mkdirs();
        preSaver(trainData,trainFolder);
        log.info("Running through the test split...");
        File testFolder = new File("testFolder");
        testFolder.mkdirs();
        preSaver(testData,testFolder);

    }

    public static void preSaver(InputSplit split, File saveFolder) throws Exception{
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
        recordReader.initialize(split, null);
        RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        dataIter.setPreProcessor(TrainedModels.VGG16NOTOP.getPreProcessor());
        //Iterate through the images
        int batch = 0;
        while (dataIter.hasNext()) {
            DataSet next = dataIter.next();
            INDArray features = next.getFeatures();
            INDArray[] outputA = vgg16NoTop.output(false,features);
            INDArray output = Nd4j.concat(0,outputA);
            int batchSize = output.size(0);
            int featureSize = output.length()/batchSize;
            //Writes the featurized images as a new dataset - we flatten the array before writing since we are going to train a simple MLP
            DataSet featurized = new DataSet(output.reshape(batchSize,featureSize),next.getLabels());
            featurized.save(new File(saveFolder,"featurized-" + batch + ".bin"));
            batch++;
        }
    }
}
