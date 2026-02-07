package dnnsg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Clint van Alten
 */
public class fnTeacher {
    public static boolean createDataset(network Teacher, double[][] Tinputs, double[][] Ttargets)
 {

    int n = Tinputs.length;      // Number of data points
    int m = Tinputs[0].length;   // Number of inputs
    int k = Ttargets[0].length;  // Number of outputs (i.e., number of classes)

    System.out.println("Number of Inputs: " + m);
    System.out.println("Number of Outputs: " + k);
    System.out.println("Total Samples: " + n + "\n");

    Map<Integer, Integer> classDistribution = new HashMap<>();
    Map<Integer, List<double[]>> classInputs = new HashMap<>();  // For standard deviation tracking

    for (int i = 0; i < n; i++) {
        for (int r = 0; r < m; r++) {
            Tinputs[i][r] = 2 * Math.random() - 1; // Inputs in range (-1,1)
        }

        fnFeedForward.feedForward(Teacher, Tinputs[i]);
        int classIndex = getOutputs(Teacher, Ttargets, i);

        classDistribution.put(classIndex, classDistribution.getOrDefault(classIndex, 0) + 1);

        // Store input vector by class for later std dev calculation
        classInputs.putIfAbsent(classIndex, new ArrayList<>());
        classInputs.get(classIndex).add(Tinputs[i].clone());
    }

    // Print class distribution
    System.out.println("\nCLASS DISTRIBUTION SUMMARY");
    System.out.println("---------------------------");
    for (int i = 0; i < k; i++) {
        int count = classDistribution.getOrDefault(i, 0);
        double percentage = (double) count / n * 100;
        System.out.printf("Class %d: %d samples (%.1f%% of total)\n", 
                        i, count, percentage);
    }

// Check if dataset is balanced

// Example: make threshold inversely related to number of inputs
double imbalanceThreshold = 1.0 / Math.sqrt(m)*5; // dynamically changes with input size

System.out.printf("Dynamic Imbalance Threshold based on input size (%d inputs): %.3f\n", m, imbalanceThreshold);
int expectedPerClass = n / k;
boolean isBalanced = true;

for (int i = 0; i < k; i++) {
    int count = classDistribution.getOrDefault(i, 0);
    double diffRatio = Math.abs(count - expectedPerClass) / (double) expectedPerClass;

    double threshold = (i > 10) ? imbalanceThreshold / 2 : imbalanceThreshold; // stricter for class > 10

    if (diffRatio > threshold) {
        System.out.printf("Class %d exceeds imbalance threshold (%.3f > %.3f)\n", i, diffRatio, threshold);
        isBalanced = false;
        break;
    }
}


if (isBalanced) {
    System.out.println("Dataset is BALANCED across output classes.");
} else {
    System.out.println("Looking for a balanced dataset");
}

return isBalanced;

}



    public static int getOutputs(network Net, double[][] T, int row) {
        layernode x = Net.outputlayernode;
        node n = x.firstnode;
        int j = 0;

        double maxVal = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;

     //   System.out.print("Raw Outputs for Data Point " + row + ": [");
        while (n != null) {
            T[row][j] = n.actvalue;
       //     System.out.print(String.format("%.3f", T[row][j]) + (n.next != null ? ", " : ""));
            
            if (T[row][j] > maxVal) { 
                maxVal = T[row][j];
                maxIndex = j;
                
               
            }

            j++;
            n = n.next;
        }
       // System.out.println("]");

        // Apply one-hot encoding
      //  System.out.print("One-Hot Encoded Output for Data Point " + row + ": [");
        for (int i = 0; i < T[row].length; i++) {
            T[row][i] = (i == maxIndex) ? 1.0 : 0.0;
       //    System.out.print((int) T[row][i] + (i < T[row].length - 1 ? ", " : ""));
        }
      //  System.out.println("]");
        
      // System.out.println("Active Class: " + maxIndex + "\n");
        
        return maxIndex;
    }

  /*  public static void printInputs(double[][] Tinputs, int row) {
        System.out.print("Inputs for Data Point " + row + ": [");
        for (int i = 0; i < Tinputs[row].length; i++) {
            System.out.print(String.format("%.3f", Tinputs[row][i]) + (i < Tinputs[row].length - 1 ? ", " : ""));
        }
        System.out.println("]");
    }

    public static void printUniqueClasses(double[][] Ttargets) {
        HashSet<Double> uniqueClasses = new HashSet<>();

        for (double[] targetRow : Ttargets) {
            for (double value : targetRow) {
                uniqueClasses.add(value);
            }
        }

        System.out.println("\nUnique Classes: " + uniqueClasses);
        System.out.println("Number of Unique Classes: " + uniqueClasses.size());
    }
    
    */
}