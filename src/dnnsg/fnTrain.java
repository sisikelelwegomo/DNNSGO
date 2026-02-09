package dnnsg;

import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import com.sun.management.OperatingSystemMXBean; // Import for CPU monitoring
import java.io.FileWriter;
import java.io.IOException;




/**
 *
 * @author Clint van Alten
 * @author Sisikelelwe Gomo
 * 
 */
public class fnTrain {
    
    
    
public static final double eta = 0.01;  
 
public static final double dropoutRate = 0.3;
 
private static final AdamOptimizer adamOptimizer = new AdamOptimizer(eta, 0.99, 0.999, 1e-8);

private static final RMSpropOptimizer rmsprop = new RMSpropOptimizer(eta, 0.99, 1e-8);

public static final double L2_LAMBDA = 0.0015;

    static double[][] Tinputs;
    
    static double[][] Ttargets;
    
    static double[][] TinputsValidate;
    
    static double[][] TtargetsValidate;
    
    static network Teacher;
    
    static network Student;
    
    public static double lastComputedLoss = 0.0;
        
    public static double lastComputedAccuracy = 0.0;
    
    public static double previousAccuracy = 0.0;
    
    public static double bestAccuracy = 0.0;
    
    public static double lastComputedTrainingLoss = 0.0;
    public static double lastComputedTrainingAccuracy = 0.0;
    public static double lastComputedValidationLoss = 0.0;
    public static double lastComputedValidationAccuracy = 0.0;
    public static double previousEpochLoss = 0.0;
    
    
public static void logHyperparameters(PrintWriter writer) {
    writer.println("=== HYPERPARAMETERS ===");
    writer.printf("Learning Rate (eta): %.6f%n", eta);
    writer.printf("Dropout Rate: %.4f%n", dropoutRate);
    writer.printf("L2 Lambda: %.6f%n", L2_LAMBDA);
    writer.printf("Adam Optimizer: beta1=%.3f, beta2=%.3f, epsilon=%.2e%n", 
                  0.99, 0.999, 1e-8);
    writer.printf("RMSprop Optimizer: decay=%.3f, epsilon=%.2e%n", 
                  0.99, 1e-8);
    writer.println("=======================");
}
    
public static void exportWeightsToCSV(network Net, String filePath) {
    
    try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
        
        writer.println("LayerType,LayerIndex,NodeIndex,EdgeIndex,Weight");
        
        int layerIndex = 0;
        
        layernode currentLayer = Net.inputlayernode;
        
        while (currentLayer != null) {
            
            String layerType = (currentLayer == Net.inputlayernode) ? "Input" : 
                    
                             (currentLayer == Net.outputlayernode) ? "Output" : "Hidden";
            
            node currentNode = currentLayer.firstnode;
            
            int nodeIndex = 0;
            
            while (currentNode != null) {
                
            
                if (currentLayer != Net.outputlayernode) {
                    
                    edge currentEdge = currentNode.firstedge;
                    
                    int edgeIndex = 0;
                    
                    while (currentEdge != null) {
                        
                        writer.printf("%s,%d,%d,%d,%.6f%n",
                                    layerType,
                                    layerIndex,
                                    nodeIndex,
                                    edgeIndex,
                                    currentEdge.weight);
                        currentEdge = currentEdge.next;
                        edgeIndex++;
                        
                    }
                    
                }
            
                else {
                    
                    layernode prevLayer = Net.inputlayernode;
                    
                 
                    while (prevLayer.next != currentLayer) {
                        
                        prevLayer = prevLayer.next;
                        
                    }
                    
               
                    node prevNode = prevLayer.firstnode;
                    
                    int edgeIndex = 0;
                    
                    while (prevNode != null) {
                        
                        edge prevEdge = prevNode.firstedge;
                        
                        while (prevEdge != null) {
                            
                       
                            if (prevEdge.target == currentNode) { 
                                
                                writer.printf("%s,%d,%d,%d,%.6f%n",
                                            layerType,
                                            layerIndex,
                                            nodeIndex,
                                            edgeIndex,
                                            prevEdge.weight);
                                
                                edgeIndex++;
                                
                            }
                            
                            prevEdge = prevEdge.next;
                            
                        }
                        
                        prevNode = prevNode.next;
                        
                    }
                    
                }
                
                currentNode = currentNode.next;
                
                nodeIndex++;
            }
            
            currentLayer = currentLayer.next;
            
            layerIndex++;
        }
        
        System.out.println("Weights saved to: " + filePath);
        
    } catch (IOException e) {
        
        System.err.println("Error saving weights: " + e.getMessage());
        
    }
    
}
  
public static void backPropagate(network Net, double[] T, double eta, boolean useDropout) {
    
    layernode x = Net.outputlayernode;
    
    if (useDropout) {
        
        updateOutputLayerWithDropout(x, T, eta);
        
    } else {
        
        updateOutputLayer(x, T, eta);
    }
    
    x = x.prev;
    
    node n;
    
    while (x.prev != null) {  
        
        n = x.firstnode;
        
        while (n != null) {
            
            if (useDropout) {
                
                updateNodeWithDropout(n, eta);
                
            } else {
                
                updateNode(n, eta);
                
            }
            
            n = n.next;
            
        }
        
        x = x.prev;
    }
    
    n = x.firstnode;
    
    while (n != null) {
        
        if (useDropout) {
            
            updateInputNodeWithDropout(n, eta);
            
        } else {
            
            updateInputNode(n, eta);
            
        }
        
        n = n.next;
        
    }
    
}

public static void backPropagate(network Net, double[] T, boolean useDropout) {
    
    layernode x = Net.outputlayernode;
    
    if (useDropout) {
        
        updateOutputLayerWithDropout(x, T);
        
    } else {
        
        updateOutputLayer(x, T);
    }
    
    x = x.prev;
    
    node n;
    
    while (x.prev != null) {  
        
        n = x.firstnode;
        
        while (n != null) {
            
            if (useDropout) {
                
                updateNodeWithDropout(n);
                
            } else {
                
                updateNode(n);
                
            }
            
            n = n.next;
            
        }
        
        x = x.prev;
    }
    
    n = x.firstnode;
    
    while (n != null) {
        
        if (useDropout) {
            
            updateInputNodeWithDropout(n);
            
        } else {
            
            updateInputNode(n);
            
        }
        
        n = n.next;
        
    }
    
}

//##################L2

// Complete L2 backpropagation methods
public static void backPropagateL2(network Net, double[] T, double eta) {
    layernode x = Net.outputlayernode;
    
    updateOutputLayerL2(x, T, eta);
    
    x = x.prev;
    node n;
    
    while (x.prev != null) {  
        n = x.firstnode;
        while (n != null) {
            updateNodeL2(n, eta);
            n = n.next;
        }
        x = x.prev;
    }
    
    n = x.firstnode;
    while (n != null) {
        updateInputNodeL2(n, eta);
        n = n.next;
    }
}

public static void backPropagateL2(network Net, double[] T) {
    layernode x = Net.outputlayernode;
    
    updateOutputLayerL2(x, T);
    
    x = x.prev;
    node n;
    
    while (x.prev != null) {  
        n = x.firstnode;
        while (n != null) {
            updateNodeL2(n);
            n = n.next;
        }
        x = x.prev;
    }
    
    n = x.firstnode;
    while (n != null) {
        updateInputNodeL2(n);
        n = n.next;
    }
}

// L2 update methods with weight updates
public static void updateOutputLayerL2(layernode x, double[] T, double eta) {
    node n = x.firstnode;
    int i = 0;
    while (n != null) {
        updateOutputNodeL2(n, T[i], eta);
        i++;
        n = n.next;
    }
}

public static void updateOutputNodeL2(node n, double target, double eta) {
    switch(n.type) {
        case 0 -> n.delta = n.actvalue - target;
        case 1 -> {
            if (n.actvalue > 0.0) n.delta = n.actvalue - target;
            else n.delta = 0.0;
        }
        case 2 -> n.delta = (n.actvalue - target) * n.actvalue * (1 - n.actvalue);
        case 3 -> n.delta = (n.actvalue - target) * (1 - (n.actvalue * n.actvalue));
        case 4 -> n.delta = n.actvalue - target;
    }
    
    n.bias -= eta * n.delta;
}

public static void updateNodeL2(node n, double eta) {
    n.delta = 0.0;
    
    for (edge e = n.firstedge; e != null; e = e.next) {
        n.delta += e.target.delta * e.weight;
        
        double gradient = e.target.delta * n.actvalue;
        gradient = Math.max(-1.0, Math.min(1.0, gradient));
        
        // Add L2 regularization: eta * L2_LAMBDA * weight
        e.weight -= eta * gradient + eta * L2_LAMBDA * e.weight;
    }

    switch(n.type) {
        case 0 -> {}
        case 1 -> {
            if (n.actvalue <= 0.0) n.delta = 0.0;
        }
        case 2 -> n.delta = n.delta * n.actvalue * (1 - n.actvalue);
        case 3 -> n.delta = n.delta * (1 - (n.actvalue * n.actvalue));
    }

    n.bias -= eta * n.delta;
}

public static void updateInputNodeL2(node n, double eta) {
    edge e = n.firstedge;
    
    while (e != null) {
        // Add L2 regularization: eta * L2_LAMBDA * weight
        e.weight -= eta * e.target.delta * n.actvalue + eta * L2_LAMBDA * e.weight;
        e = e.next;
    }
}

// L2 update methods without weight updates (for use with optimizers)
public static void updateOutputLayerL2(layernode x, double[] T) {
    node n = x.firstnode;
    int i = 0;
    while (n != null) {
        updateOutputNodeL2(n, T[i]);
        i++;
        n = n.next;
    }
}

public static void updateOutputNodeL2(node n, double target) {
    switch(n.type) {
        case 0 -> n.delta = n.actvalue - target;
        case 1 -> {
            if (n.actvalue > 0.0) n.delta = n.actvalue - target;
            else n.delta = 0.0;
        }
        case 2 -> n.delta = (n.actvalue - target) * n.actvalue * (1 - n.actvalue);
        case 3 -> n.delta = (n.actvalue - target) * (1 - (n.actvalue * n.actvalue));
        case 4 -> n.delta = n.actvalue - target;
    }
}

public static void updateNodeL2(node n) {
    n.delta = 0.0;
    for (edge e = n.firstedge; e != null; e = e.next) {
        n.delta += e.target.delta * e.weight;
    }

    switch(n.type) {
        case 0 -> {}
        case 1 -> {
            if (n.actvalue <= 0.0) n.delta = 0.0;
        }
        case 2 -> n.delta = n.delta * n.actvalue * (1 - n.actvalue);
        case 3 -> n.delta = n.delta * (1 - (n.actvalue * n.actvalue));
    }
}

public static void updateInputNodeL2(node n) {
    edge e = n.firstedge;
    while (e != null) {
        e = e.next;
    }
}

//############################################

public static void updateOutputLayerWithDropout(layernode x, double[] T, double eta) {
    
    node n = x.firstnode;
    
    int i = 0;
    
    while (n != null) {
        
        if (!n.isDroppedOut) {
            
            updateOutputNodeWithDropout(n, T[i], eta);
            
        }
        
        i++;
        
        n = n.next;
        
    }
    
}


public static void updateOutputLayerWithDropout(layernode x, double[] T) {
    
    node n = x.firstnode;
    
    int i = 0;
    
    while (n != null) {
        
        if (!n.isDroppedOut) {
            
            updateOutputNodeWithDropout(n, T[i]);
            
        }
        
        i++;
        
        n = n.next;
        
    }
    
}



public static void updateNodeWithDropout(node n, double eta) {
    
    if (n.isDroppedOut) {
        
        return; 
        
    }

    n.delta = 0.0;
    
    edge e = n.firstedge;
    
    while (e != null) {
        
        if (!e.target.isDroppedOut) { 
            
            n.delta += e.target.delta * e.weight;
            
        }
        
        e = e.next; 
        
    }
    
  
    switch(n.type) {
        
        case 0 -> {
        }
        
        case 1 -> {
            if (n.actvalue <= 0.0) n.delta = 0.0;
        }
        
        case 2 -> n.delta = n.delta * n.actvalue * (1 - n.actvalue);
        
        case 3 -> n.delta = n.delta * (1 - (n.actvalue * n.actvalue));
        
    }
    

    e = n.firstedge;
    
    while (e != null) {
        
        if (!e.target.isDroppedOut) {
            
            e.weight -= eta * e.target.delta * n.actvalue;
            
        }
        
        e = e.next;
        
    }
    
    n.bias -= eta * n.delta;
    
}
  
public static void updateNodeWithDropout(node n) {
    
    if (n.isDroppedOut) {
        
        return; 
        
    }

    n.delta = 0.0;
    
    edge e = n.firstedge;
    
    while (e != null) {
        
        if (!e.target.isDroppedOut) { 
            
            n.delta += e.target.delta * e.weight;
            
        }
        
        e = e.next; 
        
    }
    
  
    switch(n.type) {
        
        case 0 -> {
        }
        
        case 1 -> {
            if (n.actvalue <= 0.0) n.delta = 0.0;
        }
        
        case 2 -> n.delta = n.delta * n.actvalue * (1 - n.actvalue);
        
        case 3 -> n.delta = n.delta * (1 - (n.actvalue * n.actvalue));
        
    }
    

    e = n.firstedge;
    
    while (e != null) {
        
        if (!e.target.isDroppedOut) {
 
        }
        
        e = e.next;
        
    }
    
}

    public static double[] calculateLayerLearningRates( int epoch, int numepochs, int numLayers, double baseRate, double scaleFactor) {
        
        double[] learningRates = new double[numLayers];

        for (int i = 0; i < numLayers; i++) {
           
        
        learningRates[i] = baseRate * Math.pow(scaleFactor, epoch) / (i + 1);
        
        }

        return learningRates;
        
    }
    
    
    public static void backPropagateWithDynamicLearningRates(network Net, double[] T, double[] etas) {

        layernode x = Net.outputlayernode;
    
        updateOutputLayer(x, T, etas[etas.length - 1]);
    
        x = x.prev;
    
        node n;
    
        int layerIndex = etas.length - 2; 
        
      
    
        while (x.prev != null){
    
            n = x.firstnode;
        
            while (n != null){
        
                updateNode(n, etas[layerIndex]);
            
                n = n.next;
            
            }
        
            x = x.prev;
        }
    
        n = x.firstnode;
    
        while (n != null){
        
            updateInputNode(n, etas[0]); 
        
            n = n.next;
        
        }
        
    }
    
        
    
    
    public static void updateOutputLayer(layernode x, double[] T, double eta) {
    
        node n = x.firstnode;
        
        int i = 0;
        
        while (n != null) {
        
            updateOutputNode(n, T[i], eta);
            
            i++;
            
            n = n.next;        
        }
        
    }
            public static void updateOutputLayer(layernode x, double[] T) {
    
        node n = x.firstnode;
        
        int i = 0;
        
        while (n != null) {
        
            updateOutputNode(n, T[i]);
            
            i++;
            
            n = n.next;        
        }
        
    }

    public static void updateOutputNodeWithDropout(node n, double target, double eta) {
        
    if (n.isDroppedOut) return; 
    
    switch(n.type) {
        
        case 0 -> n.delta = n.actvalue - target;
        
        case 1 -> n.delta = (n.actvalue > 0.0) ? n.actvalue - target : 0.0;
        
        case 2 -> n.delta = (n.actvalue - target) * n.actvalue * (1 - n.actvalue);
        
        case 3 -> n.delta = (n.actvalue - target) * (1 - (n.actvalue * n.actvalue));
        
        case 4 -> n.delta = n.actvalue - target;
        
    }
    
    n.bias -= eta * n.delta;
 
}
        public static void updateOutputNodeWithDropout(node n, double target) {
        
    if (n.isDroppedOut) return; 
    
    switch(n.type) {
        
        case 0 -> n.delta = n.actvalue - target;
        
        case 1 -> n.delta = (n.actvalue > 0.0) ? n.actvalue - target : 0.0;
        
        case 2 -> n.delta = (n.actvalue - target) * n.actvalue * (1 - n.actvalue);
        
        case 3 -> n.delta = (n.actvalue - target) * (1 - (n.actvalue * n.actvalue));
        
        case 4 -> n.delta = n.actvalue - target;
        
    }
    
}
    public static void updateOutputNode(node n, double target, double eta) {
        
        switch(n.type) {
            
            case 0 -> n.delta = n.actvalue - target; 
                
            case 1 -> {
                if (n.actvalue > 0.0) n.delta = n.actvalue - target;
                
                else n.delta = 0.0;
        }
                
            case 2 -> n.delta = (n.actvalue - target)*n.actvalue*(1 - n.actvalue);
                
            case 3 -> n.delta = (n.actvalue - target)*(1 - (n.actvalue)*(n.actvalue));
                
            case 4 -> n.delta = n.actvalue - target;
        }
        
        n.bias -= eta*n.delta;
                
    }     
    
        public static void updateOutputNode(node n, double target) {
        
        switch(n.type) {
            
            case 0 -> n.delta = n.actvalue - target; 
                
            case 1 -> {
                if (n.actvalue > 0.0) n.delta = n.actvalue - target;
                
                else n.delta = 0.0;
        }
                
            case 2 -> n.delta = (n.actvalue - target)*n.actvalue*(1 - n.actvalue);
                
            case 3 -> n.delta = (n.actvalue - target)*(1 - (n.actvalue)*(n.actvalue));
                
            case 4 -> n.delta = n.actvalue - target;
        }
     
    }     


    public static void updateNode(node n, double eta) {
    
    n.delta = 0.0;
    
    for (edge e = n.firstedge; e != null; e = e.next) {
        
        n.delta += e.target.delta * e.weight;
        
        double gradient = e.target.delta * n.actvalue;
        
        gradient = Math.max(-1.0, Math.min(1.0, gradient));
        
        e.weight -= eta * gradient;
        
    }
    


        switch(n.type) {

            case 0 -> {
        }
                
            case 1 -> {
                if (n.actvalue <= 0.0) n.delta = 0.0;
        }
                
            case 2 -> n.delta = n.delta*n.actvalue*(1 - n.actvalue);  
                
            case 3 -> n.delta = n.delta*(1 - (n.actvalue)*(n.actvalue));
                
        }

        n.bias -= eta*n.delta;
            
    }
    
    
        public static void updateNode(node n) {
    
    n.delta = 0.0;
    
    for (edge e = n.firstedge; e != null; e = e.next) {
        
        n.delta += e.target.delta * e.weight;
        
      


    }

        switch(n.type) {

            case 0 -> {
        }
                
            case 1 -> {
                if (n.actvalue <= 0.0) n.delta = 0.0;
        }
                
            case 2 -> n.delta = n.delta*n.actvalue*(1 - n.actvalue);  
                
            case 3 -> n.delta = n.delta*(1 - (n.actvalue)*(n.actvalue));
                
        }
            
    }

public static void updateInputNode(node n, double eta) {
    
    edge e = n.firstedge;
    
    while (e != null) {
        
        e.weight -= eta * e.target.delta * n.actvalue;
        
        e = e.next;
        
    }
    
}


public static void updateInputNode(node n) {
    
    edge e = n.firstedge;
    
    while (e != null) {

        e = e.next;
        
    }
    
}


public static void updateInputNodeWithDropout(node n, double eta) {
    
    edge e = n.firstedge;
    
    while (e != null) {
        
        if (!e.target.isDroppedOut) {
            
            e.weight -= eta * e.target.delta * n.actvalue;
            
        }
        
        e = e.next;
        
    }
    
}


public static void updateInputNodeWithDropout(node n) {
    
    edge e = n.firstedge;
    
    while (e != null) {
        
        if (!e.target.isDroppedOut) {
  
        }
        
        e = e.next;
        
    }
    
}
    
    
    public static void Train(int numepochs,  double inputSize, double outputSize, double N, double SP, PrintWriter writer) throws IOException {
        
        
     // Log hyperparameters
    logHyperparameters(writer);
       
                    
        long totalTime = 0;
         
        int[] sigma = new int[Tinputs.length];
        
        for (int j = 0; j < Tinputs.length; j++) sigma[j] = j;
        
        
        boolean isBinary = (Student.outputlayernode.firstnode.next == null);
        
        Runtime runtime = Runtime.getRuntime();
        
        OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

        for (int count = 0; count < numepochs; count++) {
            
            long startTime = System.nanoTime();
            
            randomizeArray(sigma);

            for (int i = 0; i < Tinputs.length; i++)  {
    
                fnFeedForward.feedForward(Student, Tinputs[sigma[i]]);
              
                backPropagate(Student, Ttargets[sigma[i]], eta, false);
            }
            
            long endTime = System.nanoTime();
            
            long epochTime = endTime - startTime;
            
            totalTime += epochTime;
            
            double averageTime = totalTime / (double) (count + 1);
            
            double averageTimeSeconds = averageTime / 1e9;
            
            writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

            System.out.println();
            
            System.out.println("Epoch "+count);
            
            writer.println("Epoch "+count); 

            if(isBinary){
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                
                lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:");  
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);
                
                lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   
         
            } else{
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:"); 
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);
                
            }

            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            
            long freeMemory = runtime.freeMemory();

            double cpuLoad = osBean.getProcessCpuLoad() * 100;

            System.out.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
            
            System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
        
            writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            saveAverageWeights(Student, writer);
        }
        
        computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);

        computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);

        computeConfusionMatrix(Student, Tinputs, Ttargets, writer);

        computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);

        long inferStart = System.nanoTime();
    
    for (double[] TinputsValidate1 : TinputsValidate) {
        fnFeedForward.feedForward(Student, TinputsValidate1);
    }
        
        long inferEnd = System.nanoTime();

        double inferenceTime = (inferEnd - inferStart) / 1e9;

        writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

        writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);

        String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\StaticFinal\\weights5.csv";

        exportWeightsToCSV(Student, weightsPath);

        System.out.println("Weights saved to: " + weightsPath);
        
    }
    
    
        public static void TrainAdam(int numepochs,  double inputSize, double outputSize, double N, double SP, PrintWriter writer) throws IOException {

            // Log hyperparameters
            logHyperparameters(writer);
            long totalTime = 0;

            int[] sigma = new int[Tinputs.length];

            for (int j = 0; j < Tinputs.length; j++) sigma[j] = j;


            boolean isBinary = (Student.outputlayernode.firstnode.next == null);

            Runtime runtime = Runtime.getRuntime();

            OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

            for (int count = 0; count < numepochs; count++) {

                long startTime = System.nanoTime();

                randomizeArray(sigma);

                for (int i = 0; i < Tinputs.length; i++)  {

                    fnFeedForward.feedForward(Student, Tinputs[sigma[i]]);

                     backPropagate(Student, Ttargets[sigma[i]], false);
                     
                     adamOptimizer.updateWeights(Student);
                }

                long endTime = System.nanoTime();

                long epochTime = endTime - startTime;

                totalTime += epochTime;

                double averageTime = totalTime / (double) (count + 1);

                double averageTimeSeconds = averageTime / 1e9;
                
                writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

                System.out.println();

                System.out.println("Epoch "+count);

                writer.println("Epoch "+count); 

                if(isBinary){

                    lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                    
                    lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);

                    System.out.println();

                    System.out.println("Validation dataset:");

                    writer.println("Validation dataset:");  

                    lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);

                    lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   

                } else{

                    lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);

                    lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);

                    System.out.println();

                    System.out.println("Validation dataset:");

                    writer.println("Validation dataset:"); 

                    lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);

                    lastComputedTrainingAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);

                }
  
                long usedMemory = runtime.totalMemory() - runtime.freeMemory();

                long freeMemory = runtime.freeMemory();

                double cpuLoad = osBean.getProcessCpuLoad() * 100; 

                System.out.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);

                System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");

                System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");

                writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");

                writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");

                writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);

                saveAverageWeights(Student, writer);
            }

        computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);
        
        computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);

        computeConfusionMatrix(Student, Tinputs, Ttargets, writer);
        
        computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);

        long inferStart = System.nanoTime();
        
    for (double[] TinputsValidate1 : TinputsValidate) {
        fnFeedForward.feedForward(Student, TinputsValidate1); // or feedForwardWithSkip
    }
    long inferEnd = System.nanoTime();
    
    double inferenceTime = (inferEnd - inferStart) / 1e9;
    
    writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

    writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);

    String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\StaticFinal\\weights5.csv";
    
    exportWeightsToCSV(Student, weightsPath);
    
    System.out.println("Weights saved to: " + weightsPath);
    
        }

    public static void TrainRMSprop(int numepochs,  double inputSize, double outputSize, double N, double SP, PrintWriter writer) throws IOException {
    
        // Log hyperparameters
        logHyperparameters(writer);
            
        long totalTime = 0;
         
        int[] sigma = new int[Tinputs.length];
        
        for (int j = 0; j < Tinputs.length; j++) sigma[j] = j;
        
        boolean isBinary = (Student.outputlayernode.firstnode.next == null);
        
        Runtime runtime = Runtime.getRuntime();
        
        OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

        for (int count = 0; count < numepochs; count++) {
            
            long startTime = System.nanoTime();
            
            randomizeArray(sigma);

            for (int i = 0; i < Tinputs.length; i++)  {
    
                fnFeedForward.feedForward(Student, Tinputs[sigma[i]]);
                
                 backPropagate(Student, Ttargets[sigma[i]], false);
                 
                 rmsprop.updateWeights(Student);
            }
            
            long endTime = System.nanoTime();
            
            long epochTime = endTime - startTime;
            
            totalTime += epochTime;
            
            double averageTime = totalTime / (double) (count + 1);
            
            double averageTimeSeconds = averageTime / 1e9;
            
            writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

            System.out.println();
            
            System.out.println("Epoch "+count);
            
            writer.println("Epoch "+count); 
            
            if(isBinary){
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                
                lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:");  
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);
                
                lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   
      
                
                
            } else{
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:"); 
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);
                
            }
        
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            
            long freeMemory = runtime.freeMemory();
            
            double cpuLoad = osBean.getProcessCpuLoad() * 100;

            System.out.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
            
            System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
        
            writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            saveAverageWeights(Student, writer);
        }
        
            computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);

            computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);

            computeConfusionMatrix(Student, Tinputs, Ttargets, writer);

            computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);

            long inferStart = System.nanoTime();
    
            for (double[] TinputsValidate1 : TinputsValidate) {
                fnFeedForward.feedForward(Student, TinputsValidate1);
    }
                long inferEnd = System.nanoTime();

                double inferenceTime = (inferEnd - inferStart) / 1e9;

                writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

                writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);

                String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\StaticFinal\\weights5.csv";

                exportWeightsToCSV(Student, weightsPath);

                System.out.println("Weights saved to: " + weightsPath);
                    }

    
    public static void TrainDO(int numepochs,  double inputSize, double outputSize, double N, double SP, PrintWriter writer) throws IOException {
        
        
        // Log hyperparameters
        logHyperparameters(writer);
            
            
        long totalTime = 0;
         
        int[] sigma = new int[Tinputs.length];
        
        for (int j = 0; j < Tinputs.length; j++) sigma[j] = j;
        
        
        boolean isBinary = (Student.outputlayernode.firstnode.next == null);
        
        Runtime runtime = Runtime.getRuntime();
        
        OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

        for (int count = 0; count < numepochs; count++) {
            
            long startTime = System.nanoTime();
            
            randomizeArray(sigma);

            for (int i = 0; i < Tinputs.length; i++)  {
    
                fnFeedForward.feedForward(Student, Tinputs[sigma[i]], dropoutRate, true, writer);         
                
                backPropagate(Student, Ttargets[sigma[i]], eta, true);
                 
                 
            }
            
            long endTime = System.nanoTime();
            
            long epochTime = endTime - startTime;
            
            totalTime += epochTime;
            
            double averageTime = totalTime / (double) (count + 1);
            
            double averageTimeSeconds = averageTime / 1e9;
            writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

            
            System.out.println();
            
            System.out.println("Epoch "+count);
            
            writer.println("Epoch "+count); 
            
            if(isBinary){
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                
                lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:");  
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);
                
                lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   
      
            } else{
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:"); 
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);
                
            }
        
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            
            long freeMemory = runtime.freeMemory();
            
            double cpuLoad = osBean.getProcessCpuLoad() * 100;

            System.out.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
            
            System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
        
            writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            saveAverageWeights(Student, writer);
        }
        
        computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);
    
        computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);
        
        computeConfusionMatrix(Student, Tinputs, Ttargets, writer);
    
        computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);
    
        long inferStart = System.nanoTime();
        
            for (double[] TinputsValidate1 : TinputsValidate) {
                fnFeedForward.feedForward(Student, TinputsValidate1); // or feedForwardWithSkip
    }
        
        long inferEnd = System.nanoTime();
        
        double inferenceTime = (inferEnd - inferStart) / 1e9;
        
        writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

        writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);

        String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\StaticFinal\\weights5.csv";
        
        exportWeightsToCSV(Student, weightsPath);
        
        System.out.println("Weights saved to: " + weightsPath);

    }

    
    public static void TrainDOAdam(int numepochs,  double inputSize, double outputSize, double N, double SP, PrintWriter writer) throws IOException {
       
        // Log hyperparameters
        logHyperparameters(writer);
        
        long totalTime = 0;
         
        int[] sigma = new int[Tinputs.length];
        
        for (int j = 0; j < Tinputs.length; j++) sigma[j] = j;
        
        boolean isBinary = (Student.outputlayernode.firstnode.next == null);
        
        Runtime runtime = Runtime.getRuntime();
        
        OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

        for (int count = 0; count < numepochs; count++) {
            
            long startTime = System.nanoTime();
            
            randomizeArray(sigma);

            for (int i = 0; i < Tinputs.length; i++)  {
    
                fnFeedForward.feedForward(Student, Tinputs[sigma[i]], dropoutRate, true, writer);
                
                 backPropagate(Student, Ttargets[sigma[i]], true);
                 
                 adamOptimizer.updateWeights(Student);
            }
            
            long endTime = System.nanoTime();
            
            long epochTime = endTime - startTime;
            
            totalTime += epochTime;
            
            double averageTime = totalTime / (double) (count + 1);
            
            double averageTimeSeconds = averageTime / 1e9;
            
            writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

            System.out.println();
            
            System.out.println("Epoch "+count);
            
            writer.println("Epoch "+count); 
            
            if(isBinary){
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                
                lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:");  
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);
                
                lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   
      
            } else{
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:"); 
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);
                
            }
        
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            
            long freeMemory = runtime.freeMemory();
            
            double cpuLoad = osBean.getProcessCpuLoad() * 100; 

            System.out.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
            
            System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
        
            writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            saveAverageWeights(Student, writer);
        }
        
            computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);

            computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);

            computeConfusionMatrix(Student, Tinputs, Ttargets, writer);

            computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);

            long inferStart = System.nanoTime();

            for (double[] TinputsValidate1 : TinputsValidate) {
                fnFeedForward.feedForward(Student, TinputsValidate1); // or feedForwardWithSkip
    }

            long inferEnd = System.nanoTime();

            double inferenceTime = (inferEnd - inferStart) / 1e9;

            writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

            writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);

            String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\StaticFinal\\weights5.csv";

            exportWeightsToCSV(Student, weightsPath);

            System.out.println("Weights saved to: " + weightsPath);

            }

    
    
        public static void TrainDORMSprop(int numepochs,  double inputSize, double outputSize, double N, double SP, PrintWriter writer) throws IOException {
            
            
            
        // Log hyperparameters
        logHyperparameters(writer);
                    
        long totalTime = 0;
         
        int[] sigma = new int[Tinputs.length];
        
        for (int j = 0; j < Tinputs.length; j++) sigma[j] = j;
        
        boolean isBinary = (Student.outputlayernode.firstnode.next == null);
        
        Runtime runtime = Runtime.getRuntime();
        
        OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

        for (int count = 0; count < numepochs; count++) {
            
            long startTime = System.nanoTime();
            
            randomizeArray(sigma);

            for (int i = 0; i < Tinputs.length; i++)  {
    
                fnFeedForward.feedForward(Student, Tinputs[sigma[i]], dropoutRate, true, writer);
                
                backPropagate(Student, Ttargets[sigma[i]], true);
                
                rmsprop.updateWeights(Student);
            }
            
            long endTime = System.nanoTime();
            
            long epochTime = endTime - startTime;
            
            totalTime += epochTime;
            
            double averageTime = totalTime / (double) (count + 1);
            
            double averageTimeSeconds = averageTime / 1e9;
            
            writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

            System.out.println();
            
            System.out.println("Epoch "+count);
            
            writer.println("Epoch "+count); 
            
            if(isBinary){
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                
                lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:");  
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);
                
                lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   
      
            } else{
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:"); 
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);
                
            }
        
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            
            long freeMemory = runtime.freeMemory();
            
            double cpuLoad = osBean.getProcessCpuLoad() * 100;
            
            System.out.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
            
            System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
        
            writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            saveAverageWeights(Student, writer);
        }
        
    computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);
    
    computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);
        
    computeConfusionMatrix(Student, Tinputs, Ttargets, writer);
    
    computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);
    
    long inferStart = System.nanoTime();
    
    for (double[] TinputsValidate1 : TinputsValidate) {
        fnFeedForward.feedForward(Student, TinputsValidate1);
    }
    
    long inferEnd = System.nanoTime();
    
    double inferenceTime = (inferEnd - inferStart) / 1e9;
    
    writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

    writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);

    String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\StaticFinal\\weights5.csv";
    
    exportWeightsToCSV(Student, weightsPath);
    
    System.out.println("Weights saved to: " + weightsPath);
   
    }

    
    
    public static void TrainL2(int numepochs,  double inputSize, double outputSize, double N, double SP, PrintWriter writer) throws IOException {
        

                    
        // Log hyperparameters
        logHyperparameters(writer);
                    
        long totalTime = 0;
         
        int[] sigma = new int[Tinputs.length];
        
        for (int j = 0; j < Tinputs.length; j++) sigma[j] = j;
        
        boolean isBinary = (Student.outputlayernode.firstnode.next == null);
        
        Runtime runtime = Runtime.getRuntime();
        
        OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

        for (int count = 0; count < numepochs; count++) {
            
            long startTime = System.nanoTime();
            
            randomizeArray(sigma);

            for (int i = 0; i < Tinputs.length; i++)  {
    
                fnFeedForward.feedForward(Student, Tinputs[sigma[i]]);
                
                backPropagateL2(Student, Ttargets[sigma[i]], eta);
                
            }
            
            long endTime = System.nanoTime();
            
            long epochTime = endTime - startTime;
            
            totalTime += epochTime;
            
            double averageTime = totalTime / (double) (count + 1);
            
            double averageTimeSeconds = averageTime / 1e9;
            
            writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

            System.out.println();
            
            System.out.println("Epoch "+count);
            
            writer.println("Epoch "+count); 
            
            if(isBinary){
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                
                lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:");  
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);
                
                lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   

            } else{
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:"); 
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);
                
            }
        
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            
            long freeMemory = runtime.freeMemory();
            
            double cpuLoad = osBean.getProcessCpuLoad() * 100;

            System.out.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
            
            System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
        
            writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            saveAverageWeights(Student, writer);
        }
        
        computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);

        computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);

        computeConfusionMatrix(Student, Tinputs, Ttargets, writer);

        computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);

        long inferStart = System.nanoTime();
    
            for (double[] TinputsValidate1 : TinputsValidate) {
                fnFeedForward.feedForward(Student, TinputsValidate1); // or feedForwardWithSkip
    }
        
        long inferEnd = System.nanoTime();

        double inferenceTime = (inferEnd - inferStart) / 1e9;

        writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

        writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);

        String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\StaticFinal\\weights5.csv";

        exportWeightsToCSV(Student, weightsPath);

        System.out.println("Weights saved to: " + weightsPath);

        }

    
    public static void TrainL2Adam(int numepochs,  double inputSize, double outputSize, double N, double SP, PrintWriter writer) throws IOException {
        
        
          
                    
        // Log hyperparameters
        logHyperparameters(writer);
        
        long totalTime = 0;
         
        int[] sigma = new int[Tinputs.length];
        
        for (int j = 0; j < Tinputs.length; j++) sigma[j] = j;
        
        boolean isBinary = (Student.outputlayernode.firstnode.next == null);
        
        Runtime runtime = Runtime.getRuntime();
        
        OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

        for (int count = 0; count < numepochs; count++) {
            
            long startTime = System.nanoTime();
            
            randomizeArray(sigma);

            for (int i = 0; i < Tinputs.length; i++)  {
    
                fnFeedForward.feedForward(Student, Tinputs[sigma[i]]);
                
                backPropagateL2(Student, Ttargets[sigma[i]], eta);
                
                adamOptimizer.updateWeights(Student);
            }
            
            long endTime = System.nanoTime();
            
            long epochTime = endTime - startTime;
            
            totalTime += epochTime;
            
            double averageTime = totalTime / (double) (count + 1);
            
            double averageTimeSeconds = averageTime / 1e9;
            
            writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

            System.out.println();
            
            System.out.println("Epoch "+count);
            
            writer.println("Epoch "+count); 
            
            if(isBinary){
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                
                lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);

                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:");  
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);
                
                lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   
      
            } else{
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:"); 
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);
                
            }
        
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            
            long freeMemory = runtime.freeMemory();
            
            double cpuLoad = osBean.getProcessCpuLoad() * 100;

            System.out.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
            
            System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
        
            writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            saveAverageWeights(Student, writer);
        }
        
        computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);

        computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);

        computeConfusionMatrix(Student, Tinputs, Ttargets, writer);

        computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);

        long inferStart = System.nanoTime();
    
            for (double[] TinputsValidate1 : TinputsValidate) {
                fnFeedForward.feedForward(Student, TinputsValidate1);
    }

        long inferEnd = System.nanoTime();

        double inferenceTime = (inferEnd - inferStart) / 1e9;

        writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

        writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);

        String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\StaticFinal\\weights5.csv";

        exportWeightsToCSV(Student, weightsPath);

        System.out.println("Weights saved to: " + weightsPath);
        
    }

    public static void TrainL2RMSprop(int numepochs,  double inputSize, double outputSize, double N, double SP, PrintWriter writer) throws IOException {
          
            
                    
        // Log hyperparameters
        logHyperparameters(writer);
        
        long totalTime = 0;
         
        int[] sigma = new int[Tinputs.length];
        
        for (int j = 0; j < Tinputs.length; j++) sigma[j] = j;
        
        boolean isBinary = (Student.outputlayernode.firstnode.next == null);
        
        Runtime runtime = Runtime.getRuntime();
        
        OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

        for (int count = 0; count < numepochs; count++) {
            
            long startTime = System.nanoTime();
            
            randomizeArray(sigma);

            for (int i = 0; i < Tinputs.length; i++)  {
    
                fnFeedForward.feedForward(Student, Tinputs[sigma[i]]);
                
                backPropagateL2(Student, Ttargets[sigma[i]], eta);
                
                rmsprop.updateWeights(Student);
            }
            
            long endTime = System.nanoTime();
            
            long epochTime = endTime - startTime;
            
            totalTime += epochTime;
            
            double averageTime = totalTime / (double) (count + 1);
            
            double averageTimeSeconds = averageTime / 1e9;
            
            writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

            System.out.println();
            
            System.out.println("Epoch "+count);
            
            writer.println("Epoch "+count); 
            
            if(isBinary){
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                
                lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);

                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:");  
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);
                
                lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   
      
            } else{
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:"); 
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);
                
            }
        
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            
            long freeMemory = runtime.freeMemory();
            
            double cpuLoad = osBean.getProcessCpuLoad() * 100;

            System.out.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
            
            System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
        
            writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time (ns): " + averageTime);
            
            saveAverageWeights(Student, writer);
            
            }
        
            computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);

            computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);

            computeConfusionMatrix(Student, Tinputs, Ttargets, writer);

            computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);

            long inferStart = System.nanoTime();
    
            for (double[] TinputsValidate1 : TinputsValidate) {
                fnFeedForward.feedForward(Student, TinputsValidate1);
    }
            long inferEnd = System.nanoTime();

            double inferenceTime = (inferEnd - inferStart) / 1e9;

            writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

            writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);

            String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\StaticFinal\\weights5.csv";

            exportWeightsToCSV(Student, weightsPath);

            System.out.println("Weights saved to: " + weightsPath);


            }
    

/**
 * Adjusts skip connection probability based on both loss and accuracy metrics
 * 
 * @param currentSP Current skip probability
 * @param epoch Current epoch number
 * @param totalEpochs Total number of epochs
 * @param currentLoss Current epoch's loss
 * @param previousLoss Previous epoch's loss
 * @param currentAccuracy Current epoch's accuracy
 * @param previousAccuracy Previous epoch's accuracy
 * @param bestAccuracy Best accuracy seen so far
 * @return Adjusted skip probability
 */
private static double adjustSkipProbability(double currentSP, int epoch, int totalEpochs,
                                         double currentLoss, double previousLoss,
                                         double currentAccuracy, double previousAccuracy,
                                         double bestAccuracy) {
    // Base reduction based on epoch progress (gradually reduce skip connections)
    double newSP = currentSP * (1.0 - (0.5 * (double)epoch / totalEpochs));
    
    // Loss-based adjustments
    if (previousLoss > 0) {
        if (currentLoss < previousLoss) {
            // Loss improved - reduce skip connections more aggressively
            newSP *= 0.90; 
        } else if (currentLoss > previousLoss * 1.05) {
            // Loss worsened significantly - increase skip connections
            newSP *= 1.10;
        }
    }
    
    // Accuracy-based adjustments
    if (previousAccuracy > 0) {
        if (currentAccuracy > bestAccuracy) {
            // New best accuracy - reduce skip connections to encourage learning
            newSP *= 0.85;
        } else if (currentAccuracy < previousAccuracy * 0.98) {
            // Accuracy dropped - increase skip connections to stabilize
            newSP *= 1.15;
        } else if (currentAccuracy > previousAccuracy) {
            // Small improvement - slight reduction
            newSP *= 0.95;
        }
    }
    
    // Gradual reduction when accuracy plateaus
    if (Math.abs(currentAccuracy - previousAccuracy) < 0.1) {
        newSP *= 0.97;
    }
    
    // Final clamping to reasonable bounds with some randomness
    double minSP = 0.05 + (0.05 * Math.random()); // Randomness helps escape local optima
    double maxSP = 0.85 - (0.05 * Math.random());
    return Math.max(minSP, Math.min(maxSP, newSP));
}

// Helper method to calculate metrics
private static double calculateMetrics(network net, double[][] inputs, double[][] targets, 
                                     boolean isBinary, PrintWriter writer) {
    double loss = totalLoss(net, inputs, targets, writer);
    if (isBinary) {
        totalBinaryAccuracy(net, inputs, targets, writer);
    } else {
        totalAccuracy(net, inputs, targets, writer);
    }
    return loss;
}

// Helper method for final evaluation
private static void evaluateModel(network net, PrintWriter writer) {
    computePrecisionRecallF1(net, Tinputs, Ttargets, writer);
    computePrecisionRecallF1(net, TinputsValidate, TtargetsValidate, writer);
    computeConfusionMatrix(net, Tinputs, Ttargets, writer);
    computeConfusionMatrix(net, TinputsValidate, TtargetsValidate, writer);
    
    // Measure inference time
    long inferStart = System.nanoTime();
    for (double[] input : TinputsValidate) {
        fnFeedForward.feedForward(net, input);
    }
    long inferEnd = System.nanoTime();
    double inferenceTime = (inferEnd - inferStart) / 1e9;
    writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);
    writer.printf("Average Inference Time per Sample: %.6f seconds\n", 
                 inferenceTime / TinputsValidate.length);
    
    // Save final weights
    String weightsPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\Expirement1\\Dynamic\\weights_final.csv";
    exportWeightsToCSV(net, weightsPath);
    writer.println("Final weights saved to: " + weightsPath);
}

// Helper method to log system metrics
private static void logSystemMetrics(Runtime runtime, OperatingSystemMXBean osBean, 
                                   long startTime, long endTime, double averageTime, 
                                   PrintWriter writer) {
    long usedMemory = runtime.totalMemory() - runtime.freeMemory();
    long freeMemory = runtime.freeMemory();
    double cpuLoad = osBean.getProcessCpuLoad() * 100;
    
    writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
    writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + 
                  freeMemory / (1024 * 1024) + " MB");
    writer.println("Start Time: " + startTime + " End Time: " + endTime + 
                  " Average Time: " + averageTime);
}
    public static void TrainDynamic(int numepochs, double SP, double baseRate, double scaleFactor, PrintWriter writer) {


    int totalSamples = Tinputs.length;

    double[][] trainInputs = new double[totalSamples][];

    long totalTime = 0;

    int[] sigma = new int[totalSamples];  // For randomizing inputs order

    boolean isBinary = (Student.outputlayernode.firstnode.next == null);
    

    for (int j = 0; j < totalSamples; j++) {

        sigma[j] = j;
    }

    randomizeArray(sigma);
    

    for (int i = 0; i < totalSamples; i++) {
        
        trainInputs[i] = Tinputs[sigma[i]];
        
    }

    Runtime runtime = Runtime.getRuntime();
    
    OperatingSystemMXBean osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

    for (int count = 0; count < numepochs; count++) {
        
        double[] etas = calculateLayerLearningRates(count, numepochs, Student.getNumberOfLayers(), baseRate, scaleFactor);

        writer.println("Learning Rates for Each Layer:");
        
        for (int i = 0; i < etas.length; i++) {
            
            writer.printf("  Layer %d: %.6f%n", i + 1, etas[i]);
            
        }

        long startTime = System.nanoTime();
        
        for (int i = 0; i < totalSamples; i++) {
            
            if (i == 0){
            
            }

            
         
            
            backPropagateWithDynamicLearningRates(Student, Ttargets[sigma[i]], etas);
            
          
            
        }
        


        System.out.println("Dynamic Part");
	
        int initialSize = 0;

        // Network cleanup and addup
        
        System.out.println("Network addup");
        
        applyHeuristics(
        count, 
        numepochs, 
        SP,
        lastComputedTrainingLoss, 
        previousEpochLoss,
        lastComputedLoss, 
        lastComputedAccuracy, 
        lastComputedLoss, 
        lastComputedAccuracy, 
        Student,
        writer
    );

        System.out.println("Network addup");
       
    //   fnAdd.networkAddup(Student, writer);
        
        System.out.println("Network cleanup");
        
     //   fnRemove.networkRemoveup(Student, writer, numepochs, initialSize);
      //    fnRemove.networkCleanup(Student);

        long endTime = System.nanoTime();
        
        long epochTime = endTime - startTime;
        
        totalTime += epochTime;
        
        double averageTime = totalTime / (double) (count + 1);
        
        double averageTimeSeconds = averageTime / 1e9;
        writer.printf("Average Epoch Time (s): %.3f\n", averageTimeSeconds);

        // Logging epoch information
        
        System.out.println("Logging epoch information");
        
        System.out.println("\nEpoch " + count);
        
        writer.println("Epoch " + count);

        // Calculate and log losses and accuracies
        
            if(isBinary){
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets,  writer);
                lastComputedTrainingAccuracy = totalBinaryAccuracy(Student, Tinputs, Ttargets, writer);
                

                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:");  
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate,  writer);
                
                lastComputedValidationAccuracy = totalBinaryAccuracy(Student, TinputsValidate, TtargetsValidate, writer);   
      
                
                
            } else{
                
                lastComputedTrainingLoss = totalLoss(Student, Tinputs, Ttargets, writer);
                
                lastComputedTrainingAccuracy = totalAccuracy(Student, Tinputs, Ttargets, writer);
                
                
                
                System.out.println();

                System.out.println("Validation dataset:");
        
                writer.println("Validation dataset:"); 
                
                lastComputedValidationLoss = totalLoss(Student, TinputsValidate, TtargetsValidate, writer);
                
                lastComputedValidationAccuracy = totalAccuracy(Student, TinputsValidate, TtargetsValidate, writer);
                
            }
        
                    // Track memory usage
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            
            long freeMemory = runtime.freeMemory();
            
                    // CPU tracking
            double cpuLoad = osBean.getProcessCpuLoad() * 100; // Get CPU load percentage        // CPU tracking

        
            System.out.println("Start Time: " + startTime + " End Time: "  + endTime + " Average Time: " + averageTime);
            
            System.out.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
            
            System.out.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("CPU Load: " + String.format("%.2f", cpuLoad) + " %");
            
            writer.println("Memory Used: " + usedMemory / (1024 * 1024) + " MB | Free Memory: " + freeMemory / (1024 * 1024) + " MB");
        
            writer.println("Start Time: " + startTime + " End Time: " + endTime + " Average Time: " + averageTime);
            
            saveAverageWeights(Student, writer);
        }
        
            // System.out.println("Calling computePrecisionRecallF1...");
 
            computePrecisionRecallF1(Student, Tinputs, Ttargets, writer);
            computePrecisionRecallF1(Student, TinputsValidate, TtargetsValidate, writer);
        
            computeConfusionMatrix(Student, Tinputs, Ttargets, writer);
            computeConfusionMatrix(Student, TinputsValidate, TtargetsValidate, writer);


        
            // Write the formatted string to the log file
            //writer.println("Inputs: " + Arrays.deepToString(Tinputs) + ", Targets: " + Arrays.deepToString(Ttargets));
        
            long inferStart = System.nanoTime();
            for (int i = 0; i < TinputsValidate.length; i++) {
                fnFeedForward.feedForward(Student, TinputsValidate[i]); // or feedForwardWithSkip
            }
            long inferEnd = System.nanoTime();
            double inferenceTime = (inferEnd - inferStart) / 1e9;
            writer.printf("Inference Time on Validation Set: %.3f seconds\n", inferenceTime);

            writer.printf("Average Inference Time per Sample: %.6f seconds\n", inferenceTime/ TinputsValidate.length);
            
            }

    
    public static void saveAverageWeights(network Net, PrintWriter writer) {
    layernode layer = Net.inputlayernode;
    int layerIndex = 0;

    // First, count total number of layers
    int totalLayers = 0;
    layernode countLayer = Net.inputlayernode;
    while (countLayer != null) {
        totalLayers++;
        countLayer = countLayer.next;
    }

    while (layer != null) {
        double totalWeight = 0.0;
        int edgeCount = 0;

        // Count edges and weights differently for output vs. other layers
        if (layer == Net.outputlayernode) {
            if (layer.prev != null) {
                node prevNode = layer.prev.firstnode;
                while (prevNode != null) {
                    edge e = prevNode.firstedge;
                    while (e != null) {
                        totalWeight += Math.abs(e.weight);
                        edgeCount++;
                        e = e.next;
                    }
                    prevNode = prevNode.next;
                }
            }
        } else {
            // For input and hidden layers, count outgoing edges
            node currentNode = layer.firstnode;
            while (currentNode != null) {
                edge e = currentNode.firstedge;
                while (e != null) {
                    totalWeight += Math.abs(e.weight);
                    edgeCount++;
                    e = e.next;
                }
                currentNode = currentNode.next;
            }
        }

        double avgWeight = (edgeCount > 0) ? (totalWeight / edgeCount) : 0.0;

        // Determine layer label
        String label;
        if (layerIndex == 0) {
            label = "Input Layer";
        } else if (layerIndex == totalLayers - 1) {
            label = "Output Layer";
        } else {
            label = "Hidden Layer " + layerIndex;
        }

        writer.printf("%s: Average Weight = %.6f | Edge Count = %d%n", label, avgWeight, edgeCount);

        // Count nodes with no outgoing edges in the output layer
        if (label.equals("Output Layer")) {
            int noEdgeNodes = 0;
            node n = layer.firstnode;
            while (n != null) {
                if (n.firstedge == null) {
                    noEdgeNodes++;
                }
                n = n.next;
            }
            writer.printf("Output Layer: Nodes with no outgoing edges = %d%n", noEdgeNodes);
        }

        layer = layer.next;
        layerIndex++;
    }
}


public static void saveAverageWeightsDebug(network Net, PrintWriter writer) {
    layernode layer = Net.inputlayernode;
    int layerIndex = 0;

    // First, count total number of layers
    int totalLayers = 0;
    layernode countLayer = Net.inputlayernode;
    while (countLayer != null) {
        totalLayers++;
        countLayer = countLayer.next;
    }

    while (layer != null) {
        double totalWeight = 0.0;
        int edgeCount = 0;

        // Count edges and weights differently for output vs. other layers
        if (layer == Net.outputlayernode) {
            if (layer.prev != null) {
                node prevNode = layer.prev.firstnode;
                while (prevNode != null) {
                    edge e = prevNode.firstedge;
                    while (e != null) {
                        totalWeight += Math.abs(e.weight);
                        edgeCount++;
                        e = e.next;
                    }
                    prevNode = prevNode.next;
                }
            }
        } else {
            // For input and hidden layers, count outgoing edges
            node currentNode = layer.firstnode;
            while (currentNode != null) {
                edge e = currentNode.firstedge;
                while (e != null) {
                    totalWeight += Math.abs(e.weight);
                    edgeCount++;
                    e = e.next;
                }
                currentNode = currentNode.next;
            }
        }

        double avgWeight = (edgeCount > 0) ? (totalWeight / edgeCount) : 0.0;

        // Determine layer label
        String label;
        if (layerIndex == 0) {
            label = "Input Layer";
        } else if (layerIndex == totalLayers - 1) {
            label = "Output Layer";
        } else {
            label = "Hidden Layer " + layerIndex;
        }

        writer.printf("%s: Average Weight = %.6f | Edge Count = %d%n", label, avgWeight, edgeCount);

        // Count nodes with no outgoing edges in the output layer
        if (label.equals("Output Layer")) {
            int noEdgeNodes = 0;
            node n = layer.firstnode;
            while (n != null) {
                if (n.firstedge == null) {
                    noEdgeNodes++;
                }
                n = n.next;
            }
            writer.printf("Output Layer: Nodes with no outgoing edges = %d%n", noEdgeNodes);
        }

        layer = layer.next;
        layerIndex++;
    }
}

    
    public static void computePrecisionRecallF1(network Net, double[][] Inputs, double[][] Targets, PrintWriter writer) {
        
        int numClasses = Targets[0].length; // Number of classes
    
        int[] truePositives = new int[numClasses];
    
        int[] falsePositives = new int[numClasses];
    
        int[] falseNegatives = new int[numClasses];
    
        int[] totalSamplesPerClass = new int[numClasses]; // Needed for weighted averaging

        for (int i = 0; i < Inputs.length; i++) {
        
        fnFeedForward.feedForward(Net, Inputs[i]);

        // Get predicted class (argmax)
        
        int predictedClass = getPredictedClass(Net);

        // Get actual class (assuming one-hot encoding)
        
        int actualClass = getActualClass(Targets[i]);

        totalSamplesPerClass[actualClass]++; // Track occurrences of each class

        if (predictedClass == actualClass) {
            
            truePositives[actualClass]++;
            
        } else {
            
            falsePositives[predictedClass]++;
            
            falseNegatives[actualClass]++;
            
        }
        
    }

    // Compute precision, recall, and F1-score per class
    
    double macroPrecision = 0.0, macroRecall = 0.0, macroF1 = 0.0;
    
    double weightedPrecision = 0.0, weightedRecall = 0.0, weightedF1 = 0.0;
    
    int totalSamples = Inputs.length;

    for (int c = 0; c < numClasses; c++) {
        
        double precision = (truePositives[c] + falsePositives[c] > 0) ? 
                
            (double) truePositives[c] / (truePositives[c] + falsePositives[c]) : 0.0;
        
        double recall = (truePositives[c] + falseNegatives[c] > 0) ? 
                
            (double) truePositives[c] / (truePositives[c] + falseNegatives[c]) : 0.0;
        
        double f1Score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0.0;

        macroPrecision += precision;
        
        macroRecall += recall;
        
        macroF1 += f1Score;

        weightedPrecision += precision * totalSamplesPerClass[c] / totalSamples;
        
        weightedRecall += recall * totalSamplesPerClass[c] / totalSamples;
        
        weightedF1 += f1Score * totalSamplesPerClass[c] / totalSamples;

        writer.printf("Class %d -> Precision: %.4f, Recall: %.4f, F1: %.4f%n", c, precision, recall, f1Score);
    }

    // Compute macro-averaged metrics
    macroPrecision /= numClasses;
    
    macroRecall /= numClasses;
            
    macroF1 /= numClasses;

    writer.printf("\nMacro-Averaged Precision: %.4f%n", macroPrecision);
    
    writer.printf("Macro-Averaged Recall: %.4f%n", macroRecall);
    
    writer.printf("Macro-Averaged F1 Score: %.4f%n", macroF1);

    writer.printf("\nWeighted Precision: %.4f%n", weightedPrecision);
    
    writer.printf("Weighted Recall: %.4f%n", weightedRecall);
    
    writer.printf("Weighted F1 Score: %.4f%n", weightedF1);

    System.out.println("Macro Precision: " + macroPrecision);
    
    System.out.println("Macro Recall: " + macroRecall);
    
    System.out.println("Macro F1 Score: " + macroF1);

    System.out.println("Weighted Precision: " + weightedPrecision);
    
    System.out.println("Weighted Recall: " + weightedRecall);
    
    System.out.println("Weighted F1 Score: " + weightedF1);
}


    
    // Helper method to get predicted class (argmax of softmax outputs)
    private static int getPredictedClass(network Net) {
    
        layernode outputLayer = Net.outputlayernode;
    
        node n = outputLayer.firstnode;

        // Store softmax outputs
    
        double[] softmaxOutput = new double[getNumOutputClasses(Net)];
    
        int classIndex = 0;

        // Collect softmax values
    
        while (n != null) {
        
            softmaxOutput[classIndex] = n.actvalue;  // Store probability from softmax layer
        
            n = n.next;
        
            classIndex++;
        }

    // Print softmax values for debugging
    
    // System.out.println(" Output: " + Arrays.toString(softmaxOutput));

    // Find the class with the highest probability
    
        int predictedClass = 0;
    
        double maxActivation = softmaxOutput[0];

        for (int i = 1; i < softmaxOutput.length; i++) {
        
            if (softmaxOutput[i] > maxActivation) {
            
            maxActivation = softmaxOutput[i];
            
            predictedClass = i;
            
        }
    }

    return predictedClass;
    
    }

    // Helper function to get number of output classes
    private static int getNumOutputClasses(network Net) {
    
        int count = 0;
    
        node n = Net.outputlayernode.firstnode;
    
    while (n != null) {
        
        count++;
        
        n = n.next;
    }
    
    return count;
    
    }


    // Helper method to get actual class (from one-hot encoded target)
    private static int getActualClass(double[] target) {
    
    for (int i = 0; i < target.length; i++) {
        
        if (target[i] == 1.0) {
            
            return i;
            
        }
        
    }
    
    return -1; // Should not happen in a valid dataset
    
    }


    public static void computeConfusionMatrix(network Net, double[][] Inputs, double[][] Targets, PrintWriter writer) {
    
        int numClasses = Targets[0].length; // Number of classes
    
        int[][] confusionMatrix = new int[numClasses][numClasses]; // Initialize matrix

    for (int i = 0; i < Inputs.length; i++) {
        
        fnFeedForward.feedForward(Net, Inputs[i]);

        // Get predicted and actual class indices
        
        int predictedClass = getPredictedClass(Net);
        
        int actualClass = getActualClass(Targets[i]);

        // Update confusion matrix
        
        confusionMatrix[actualClass][predictedClass]++;
        
    }

    // Print confusion matrix
    
        System.out.println("\nConfusion Matrix:");
    
        writer.println("\nConfusion Matrix:");

    for (int i = 0; i < numClasses; i++) {
        
        for (int j = 0; j < numClasses; j++) {
            
            System.out.printf("%5d ", confusionMatrix[i][j]);
            
            writer.printf("%5d ", confusionMatrix[i][j]);
            
        }
        
        System.out.println();
        
        writer.println();
        
    }
    
    }


    
   private static double applyHeuristics(
    int epoch, int numepochs, 
    double skipPercentage,
    double epochLoss, 
    double previousEpochLoss,
    double DynamicLoss, 
    double DynamicAccuracy, 
    double previousLoss, 
    double previousAccuracy, 
    network studentNet,
    PrintWriter writer
) {

    writer.println("----- Heuristic Adjustment for Epoch " + epoch + " -----");

    // Epoch-based heuristic: gradually reduce skip connections
    double epochBasedLimit = Math.max(0.1, 1.0 - (double) epoch / numepochs);
    if (skipPercentage > epochBasedLimit) {
        writer.println("  Epoch progression adjustment: limiting skip percentage to " + epochBasedLimit);
        skipPercentage = epochBasedLimit;
    }

    // Epoch loss-based heuristic
    if (previousEpochLoss - epochLoss > 0.01) {
        writer.println("  Epoch loss improved. Reducing skip percentage by 0.1.");
        skipPercentage -= 0.1;
    } else if (Math.abs(previousEpochLoss - epochLoss) < 0.001) {
        writer.println("  Epoch loss stabilized. Slightly reducing skip percentage by 0.05.");
        skipPercentage -= 0.05;
    }

    // General loss trend heuristic
    if (previousLoss - DynamicLoss > 0.01) {
        writer.println("  Loss decreased. Reducing skip percentage by 0.1.");
        skipPercentage -= 0.1;
    } else if (Math.abs(previousLoss - DynamicLoss) < 0.001) {
        writer.println("  Loss stabilized. Slightly reducing skip percentage by 0.05.");
        skipPercentage -= 0.05;
    } else if (DynamicLoss > previousLoss) {
        writer.println("  Loss increased. Increasing skip percentage by 0.1.");
        skipPercentage += 0.1;
    }

    // Accuracy-based heuristic
    if (DynamicAccuracy > previousAccuracy + 0.01) {
        writer.println("  Accuracy improved. Reducing skip percentage by 0.05.");
        skipPercentage -= 0.05;
    } else if (DynamicAccuracy < previousAccuracy) {
        writer.println("  Accuracy dropped. Increasing skip percentage by 0.1.");
        skipPercentage += 0.1;
    }

    // Gradient-based heuristics
    if (detectedVanishingGradients(studentNet)) {
        writer.println("  Vanishing gradients detected. Increasing skip percentage by 0.1.");
        skipPercentage += 0.1;
    } else if (detectedExplodingGradients(studentNet)) {
        writer.println("  Exploding gradients detected. Reducing skip percentage by 0.1.");
        skipPercentage -= 0.1;
    }

    // Clamp final value
    skipPercentage = Math.max(0.0, Math.min(1.0, skipPercentage));
    writer.println("  Final adjusted skip percentage: " + skipPercentage);
    writer.flush();

    return skipPercentage;
}


    public static boolean detectedVanishingGradients(network Net) {
        double threshold = 1e-5;
        layernode current = Net.inputlayernode;
        while (current != null) {
            node n = current.firstnode;
            while (n != null) {
                if (Math.abs(n.delta) > threshold) return false;
                n = n.next;
            }
            current = current.next;
        }
        return true; // All gradients are below the threshold
    }

    public static boolean detectedExplodingGradients(network Net) {
        double threshold = 1e3; // Example threshold for exploding gradients
        layernode current = Net.inputlayernode;
        while (current != null) {
            node n = current.firstnode;
            while (n != null) {
                if (Math.abs(n.delta) > threshold) return true;
                n = n.next;
            }
            current = current.next;
        }
        return false;
    }


    public static double singleLoss(network Net, double[] A, double[] T) {
    
    // Calculates the sum-of-squares loss on input A with respect to target T.
    
    fnFeedForward.feedForward(Net, A);
    
    node n = Net.outputlayernode.firstnode;
    
    double loss = 0.0;
    
    for (int j = 0; j < T.length; j++) {
        
        loss += Math.pow(n.actvalue - T[j], 2);
        
        n = n.next;
        
    }
    
    return loss;
    
    
    
    }
    
    
    public static double singleLossL2(network Net, double[] A, double[] T) {
    fnFeedForward.feedForward(Net, A);
    node n = Net.outputlayernode.firstnode;
    double loss = 0.0;
    
    // Standard loss
    for (int j = 0; j < T.length; j++) {
        loss += Math.pow(n.actvalue - T[j], 2);
        n = n.next;
    }
    
    // Add L2 regularization term
    double l2Penalty = 0.0;
    layernode layer = Net.inputlayernode;
    while (layer != null) {
        node currentNode = layer.firstnode;
        while (currentNode != null) {
            edge e = currentNode.firstedge;
            while (e != null) {
                l2Penalty += e.weight * e.weight;
                e = e.next;
            }
            currentNode = currentNode.next;
        }
        layer = layer.next;
    }
    
    loss += (L2_LAMBDA / 2) * l2Penalty;
    
    return loss;
}
    

    public static double totalLoss(network Net, double[][] Inputs, double[][] Targets, PrintWriter writer) {
    
    double totalloss = 0.0;
    
    for (int i = 0; i < Inputs.length; i++) {
        
        totalloss += singleLoss(Net, Inputs[i], Targets[i]);
        
    }
    
    // For a sum-of-squares loss, it's common to divide by 2.
    
    totalloss = totalloss / Inputs.length;
    
    lastComputedLoss = totalloss;  // Store the computed loss

    System.out.println("    Loss on Dataset = " + totalloss);
    
    writer.println("    Loss on Dataset = " + totalloss);
    
    return totalloss;
    
    }


    public static int singleAccuracy(network Net, double[] A, double[] T) {
    
    // Perform forward pass
    
        fnFeedForward.feedForward(Net, A);

        node n = Net.outputlayernode.firstnode;

    // Find the predicted class (argmax)
    
        int predictedClass = 0;
    
        double maxProbability = n.actvalue;

    for (int j = 1; j < T.length; j++) {
        
        n = n.next;
        
        if (n.actvalue > maxProbability) {
            
            maxProbability = n.actvalue;
            
            predictedClass = j;
        }
        
    }

    // Find the actual class (assuming one-hot encoding)
    
        int actualClass = 0;
    
    for (int j = 0; j < T.length; j++) {
        
        if (T[j] == 1.0) {
            
            actualClass = j;
            
            break;
        }
        
    }

    // Return 1 if correct, 0 otherwise
    
    return (predictedClass == actualClass) ? 1 : 0;
}


    public static double totalAccuracy(network Net, double[][] Inputs, double[][] Targets, PrintWriter writer) {
    
        double correctCount = 0.0;
    
    for (int i = 0; i < Inputs.length; i++) {
        
        correctCount += singleAccuracy(Net, Inputs[i], Targets[i]);
        
    }
    // Convert the count to a percentage
    
        double percentAccuracy = (correctCount / Inputs.length) * 100.0;
    
        lastComputedAccuracy = percentAccuracy; // Store computed accuracy

        System.out.println("    Accuracy on Dataset = " + percentAccuracy + " %");
    
        writer.println("    Accuracy on Dataset = " + percentAccuracy + " %");

    return percentAccuracy;
    }


    public static int binaryAccuracy(network Net, double[] A, double[] T) {
    
        // For binary classification: Use a threshold of 0.5 to decide the class.
    
        fnFeedForward.feedForward(Net, A);
    
        node n = Net.outputlayernode.firstnode;
    
        double maxval = n.actvalue;
    
        int predLabel = (maxval >= 0.5) ? 1 : 0;
    
        int actualLabel = (T[0] >= 0.5) ? 1 : 0;
    
    return (predLabel == actualLabel) ? 1 : 0;
    
    }

    public static double totalBinaryAccuracy(network Net, double[][] Inputs, double[][] Targets, PrintWriter writer) {
    
        double correctCount = 0.0;
    
    for (int i = 0; i < Inputs.length; i++) {
        
        correctCount += binaryAccuracy(Net, Inputs[i], Targets[i]);
        
    }
    
    double percentAccuracy = (correctCount / Inputs.length) * 100;
    
        System.out.println("    Binary Accuracy on Dataset = " + percentAccuracy + " %");
    
        writer.println("    Binary Accuracy on Dataset = " + percentAccuracy + " %");
    
    return percentAccuracy;
    }
    
    public static void randomizeArray(int[] A) {

        int temp;
        
        int randomindex;   
    
        for (int i = 0; i < A.length - 1; i++) {
    
            randomindex =  i + (int)(Math.random() * (A.length - i)); 
   
            temp = A[i];
            
            A[i] = A[randomindex];
            
            A[randomindex] = temp;
            
        }
        
    }  
    
}