import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;

import java.io.IOException;

/**
 * University of Central Florida
 * CAP4630 Artificial Intelligence - Spring 2019
 * Multi-Class Vector Perceptron Classifier
 * Author(s): Yashwant Bhosale and William Cromar
 */
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
        System.out.println("University of Central Florida");
        System.out.println("CAP4630 Artificial Intelligence - Spring 2019");
        System.out.println("Multi-Class Perceptron Classifier");
        System.out.println("Author(s): Yashwant Bhosale and William Cromar");
        System.out.println();

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
        String s = new String();
        s += ("Source file: " + fileName + "\n");
        s +=("Training epoch limit   : " + maxNumEpochs + "\n");
        s +=("Actual training epochs : " + layer.actualEpochs + "\n");
        s +=("Total # weight updates : " + layer.weightUpdates + "\n");
        s += layer.printWeights();
        return s;
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
            int argMaxIndex = 0;

            Matrix activations = weights.times(input);
            argMax = activations.get(0, 0);

            for(int i = 0; i < length; i ++) {

                if (argMax < activations.get(i, 0)) {
                    argMax = activations.get(i, 0);
                    argMaxIndex = i;
                }
            }

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

        public String printWeights(){
            String s = "\nFinal weights:\n\n";
            double[][] weightsCopy = weights.getArrayCopy();

            for(int i = 0; i < weightsCopy.length; i++ ){
                s += String.format("Class %d: ", i);
                for(int j = 0; j < weightsCopy[i].length; j++){
                    s += String.format("%7.3f ", weightsCopy[i][j]);
                }
                s += "\n";
            }
            return s;
        }


        public void train(Instances instances){
            int classification;
            boolean weightUpdate = false;
            int epoch;

            for(epoch = 0; epoch < maxEpochs; epoch++) {
                System.out.printf("Epoch   %d ", epoch);

                StringBuilder sb = new StringBuilder();
                int cnt = 0;

                weightUpdate = false;
                for (Instance in : instances) {

                    classification = classifyInstance(in);

                    // If the classification is wrong, then you have to add the input values to the weights of the correct
                    // index and subtract them from the wrong index
//                  // this might be wrong
                    if (in.classValue() != classification) {
                        sb.append("0");
                        cnt++;

                        weightUpdate = true;
                        weightUpdates++;
                        for (int i = 0; i < in.numAttributes() - 1; i++) {
                            weights.getArray()[classification][i] -= in.value(i);
                            weights.getArray()[(int) in.classValue()][i] += in.value(i);
                        }
                        weights.getArray()[classification][in.numAttributes() - 1] -= 1;
                        weights.getArray()[(int) in.classValue()][in.numAttributes() - 1] += 1;
                    } else {
                        sb.append("1");
                    }
                }

                System.out.printf("[%2d] ", cnt);
                System.out.println(sb);

                if(weightUpdate == false)
                    break;
            }

            actualEpochs = epoch + 1;

            return;
        }

    }

}


