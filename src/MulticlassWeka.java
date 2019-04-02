import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;

/**
 * University of Central Florida
 * CAP4630 Artificial Intelligence
 * Multi-Class Vector Perceptron Classifier Driver Class
 * Author:  Dr. Demetrios Glinos
 */
public class MulticlassWeka {
    private static final DecimalFormat DF = new DecimalFormat("0.00");

    public static BufferedReader findDataFile(String filename) {

        BufferedReader inputReader = null;
        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }
        return inputReader;
    }

    public static Evaluation classify(
        Classifier model, Instances dataSet ) throws Exception {
        Evaluation evaluation = new Evaluation(dataSet);
        model.buildClassifier(dataSet);

        evaluation.evaluateModel(model, dataSet);
        System.out.println( evaluation.toSummaryString("\nResults:\n", true) );

        return evaluation;
    }

    public static double calculateAccuracy(ArrayList<Prediction> predictions) {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.get(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }
        return 100 * correct / predictions.size();
    }

    public static void main(String[] args) throws Exception {

        // Load data file of instances
        BufferedReader datafile = findDataFile(args[0]);
        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);

        // Instantiate perceptron classifier with with command arguments
        String[] options = new String[2];
        options[0] = args[0];   // input file name
        options[1] = args[1];   // number of training epochs
        Classifier model = new MulticlassPerceptron( options );

        // Run classifier and report results
        Evaluation validation = classify(model, data);
        ArrayList<Prediction> predictions = validation.predictions();

        // Final score
        double accuracy = calculateAccuracy(predictions);
        System.out.println("Accuracy of "
                + model.getClass().getSimpleName() + ": "
                + DF.format(accuracy) + " %\n");

        // Echo experiment setup and final weights
        System.out.println(model.toString() + "\n\n");
    }
}
         
