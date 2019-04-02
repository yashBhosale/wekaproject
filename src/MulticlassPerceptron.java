import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math.*;

public class MulticlassPerceptron implements Classifier{


    private MCP layer;
    public int maxNumEpochs;
    public String fileName;

    public MulticlassPerceptron(String[] options) throws IOException {
        fileName = options[0];
        maxNumEpochs = Integer.parseInt(options[1]);
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        layer = new MCP(instances.numClasses(), maxNumEpochs);
        layer.train(instances);

        return;
    }

    //Classify the instance and then return the index of the class (0 to numClasses-1)
    int predict(Instance instance){
        return layer.classifyInstance(instance);
    }

    //----Don't worry about these three methods.
        @Override
        public double[] distributionForInstance(Instance instance) throws Exception {
            double[] result = new double [instance.numClasses()];
            result[predict(instance)] = 1;
            return result;
        }

        @Override
        public Capabilities getCapabilities() {
            return null;
        }

//      Note to self: is it better to just have the preprocessing
        @Override
        public double classifyInstance(Instance instance) throws Exception {
            return 0;
        }
//----

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append("Source file: " + fileName + "/n");
        s.append("Training epoch limit : " + maxNumEpochs);
        s.append("actual training epochs : " + layer.actualEpochs);
        s.append("total number weight updates : " + layer.weightUpdates);

        return super.toString();
    }


    private class MCP{
        Matrix weights;
        float length;
        int maxEpochs, actualEpochs, weightUpdates;

        public MCP(int length, int maxEpochs){
            this.maxEpochs = maxEpochs;
            this.length = length;
            this.actualEpochs = 0;
            this.weightUpdates = 0;
            this.weights = new Matrix(length, length + 1);
        }


        public int classifyInstance(Instance in){
            //add a 1 to the end and make it an nx1 matrix
            Matrix instance = preprocessInstance(in);
            return classify(instance);
        }

        public int classify(Matrix input){
            double argMax;
            int argMaxIndex =0;

            Matrix activations = weights.times(input);
            argMax = activations.get(0, 0);

            for(int i = 0; i < length; i ++) {

                if (argMax < activations.get(i, 0)) {
                    System.out.println(activations.get(i, 0) + " > " + argMax);
                    argMax = activations.get(i, 0);
                    argMaxIndex = i;
                }
            }
            System.out.println("argmax index is " + argMaxIndex);
            return argMaxIndex;
        }

//      Just tacking on an extra 1 to the end of the instance and making it into a straight array
        public Matrix preprocessInstance(Instance in){
//          Moving the data from the instance into an array with an extra space for the bias input
//          We're making it of size numAttributes because the class label is stored as an attribute
//          but we obviously don't need it for internal mechanics

//          TODO: make sure numAttributes works (lol)
            double[] inputVals = new double[in.numAttributes()], inArr = in.toDoubleArray();

            for(int i = 0; i < in.numAttributes()-1;i++)
                inputVals[i] = inArr[i];

            //adding the bias input to the end of the input array and converting it into a Matrix
            inputVals[in.numAttributes()-1] = 1;

            return new Matrix(inputVals, 1).transpose();
        }

        public void train(Instances instances){
            int classification;
            boolean noWeightUpdate = true;
            int epoch;
            for(epoch= 0; epoch < maxEpochs || noWeightUpdate; epoch++) {
                noWeightUpdate = true;
                for (Instance in : instances) {

                    classification = classifyInstance(in);

                    // If the classification is wrong, then you have to add the input values to the weights of the correct
                    // index and subtract them from the wrong index
                    if (in.value(in.classIndex()) != classification) {
                        noWeightUpdate = false;
                        for (int i = 0; i < in.numAttributes(); i++) {
                            weights.getArray()[classification][i] -= in.value(i);
                            weights.getArray()[(int) in.value(in.numAttributes() - 1)][i] += in.value(i);
                        }
                    }
                }
            }
            actualEpochs = epoch + 1;

            return;
        }

    }

}


