package org.deeplearning4j.examples.modelimport.trainedmodels;

import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.List;

/**
 *
 * This example demonstrates how to import VGG16 into DL4J via Keras weights and json configs.
 * //FIXME - Not uploaded to remote
 * The required H5 and json configs will be downloaded to ~/.dl4j/trainedmodels/vgg16 during the first run.
 * Note the H5 file is ~500MB.
 *
 * All images in a given directory are run through VGG16 and predictions reported.
 *
 * //FIXME
 * Citation:
 *
 * @author susaneraly
 */
public class PredictWithVGG16 {

    public static final String IMAGE_DIR = "/Users/susaneraly/SKYMIND/kerasImport/tests/imageNet/imagesMakeShift";
    public static final File parentDir = new File(IMAGE_DIR);
    public static final int batchSize = 2;

    public static void main(String [] args) throws Exception {

        //Helper for trained deep learning models
        TrainedModelHelper helper = new TrainedModelHelper(TrainedModels.VGG16);
        helper.setPathToH5("/Users/susaneraly/SKYMIND/kerasImport/VGG16/saved/vgg16New.h5");
        helper.setPathToJSON("/Users/susaneraly/SKYMIND/kerasImport/VGG16/saved/vgg16New.json");

        //Dataset iterator using an image record reader
        ImageRecordReader rr = new ImageRecordReader(224,224,3);
        rr.initialize(new FileSplit(parentDir));
        RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(rr,batchSize);
        dataIter.setCollectMetaData(true);

        //Attach the VGG16 specific preprocessor to the dataset iterator for the mean shifting required
        DataSetPreProcessor preProcessor = TrainedModels.VGG16.getPreProcessor();
        dataIter.setPreProcessor(preProcessor);

        //Load the model into dl4j
        ComputationGraph vgg16 = helper.loadModel();

        //Iterate through the images
        while (dataIter.hasNext()) {
            //prediction array
            DataSet next = dataIter.next();
            INDArray features = next.getFeatures();
            INDArray[] outputA = vgg16.output(false,features);
            INDArray output = Nd4j.concat(0,outputA);

            //print top 5 predictions for each image in the dataset
            List<RecordMetaData> trainMetaData = next.getExampleMetaData(RecordMetaData.class);
            int batch = 0;
            for(RecordMetaData recordMetaData : trainMetaData){
                System.out.println(recordMetaData.getLocation());
                System.out.println(TrainedModels.VGG16.decodePredictions(output.getRow(batch)));
                batch++;
            }

        }

    }
}
