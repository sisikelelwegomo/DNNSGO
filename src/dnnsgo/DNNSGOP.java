
package dnnsgo;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class DNNSGOP {

    // Parkinson's dataset config
    static final int INPUT_SIZE = 22;
    static final int OUTPUT_SIZE = 2;
    static final String PARKINSONS_CSV_PATH = "C:\\Users\\jimmy\\Documents\\Real Data Thesis work\\Real Data\\Datasets\\data\\processed\\parkinsons\\parkinsons.csv";
    static final int NUM_EPOCHS = 100;
    static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    static final int[] N_values = {5, 15, 30, 60, 100};
    static final double[] skipPercentages = {0.00, 0.25, 0.50, 0.75, 1.00};

    static class HyperparameterSet {
        String name;
        double learningRate;
        double dropoutRate;
        double l2Lambda;
        double adamBeta1;
        double adamBeta2;
        double rmspropDecay;
        HyperparameterSet(String name, double lr, double dropout, double l2, double beta1, double beta2, double decay) {
            this.name = name;
            this.learningRate = lr;
            this.dropoutRate = dropout;
            this.l2Lambda = l2;
            this.adamBeta1 = beta1;
            this.adamBeta2 = beta2;
            this.rmspropDecay = decay;
        }
    }

    static class TrainingConfig {
        double[][] inputs;
        double[][] targets;
        double[][] inputsValidate;
        double[][] targetsValidate;
        int input;
        int output;
        int N;
        double skipPercentage;
        HyperparameterSet hp;
        String methodName;
        String baseDirectoryPath;
        TrainingConfig(double[][] inputs, double[][] targets, double[][] inputsValidate, double[][] targetsValidate,
                      int input, int output, int N, double skipPercentage, HyperparameterSet hp, String methodName, String baseDirectoryPath) {
            this.inputs = inputs;
            this.targets = targets;
            this.inputsValidate = inputsValidate;
            this.targetsValidate = targetsValidate;
            this.input = input;
            this.output = output;
            this.N = N;
            this.skipPercentage = skipPercentage;
            this.hp = hp;
            this.methodName = methodName;
            this.baseDirectoryPath = baseDirectoryPath;
        }
    }

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("PARKINSON'S DATASET MULTITHREADED TRAINING");
        System.out.println("Available processors: " + NUM_THREADS);
        System.out.println("========================================\n");

        // Load Parkinson's data
        ParkinsonsDataLoader dataLoader = new ParkinsonsDataLoader();
        try {
            dataLoader.load(PARKINSONS_CSV_PATH, INPUT_SIZE, OUTPUT_SIZE);
        } catch (IOException e) {
            System.err.println("Failed to load Parkinson's dataset: " + e.getMessage());
            return;
        }

        // Define hyperparameter sets
        HyperparameterSet[] experiments = {
            new HyperparameterSet("Exp1_Baseline", 0.001, 0.2, 0.0005, 0.99, 0.999, 0.99),
            new HyperparameterSet("Exp2_HighLR", 0.01, 0.3, 0.0015, 0.99, 0.999, 0.99),
            new HyperparameterSet("Exp3_LowLR", 0.0005, 0.1, 0.0002, 0.99, 0.999, 0.99),
            new HyperparameterSet("Exp4_HighDropout", 0.002, 0.4, 0.0025, 0.99, 0.999, 0.99),
            new HyperparameterSet("Exp5_LowDropout", 0.0015, 0.15, 0.0007, 0.99, 0.999, 0.99),
            new HyperparameterSet("Exp6_HighL2", 0.005, 0.25,0.0010, 0.99, 0.999, 0.99),
            new HyperparameterSet("Exp7_LowL2", 0.0008, 0.05, 0.0003, 0.99, 0.999, 0.99),
            new HyperparameterSet("Exp8_NoSkip", 0.003, 0.35, 0.0009, 0.99, 0.999, 0.99)
        };

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        AtomicInteger completedTasks = new AtomicInteger(0);
        AtomicInteger totalTasks = new AtomicInteger(0);

        for (double skipPercentage : skipPercentages) {
            for (HyperparameterSet hp : experiments) {
                if (hp.name.equals("Exp8_NoSkip") && skipPercentage > 0) continue;
                for (int N : N_values) {
                    String baseDirectoryPath = String.format(
                        "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSGOP\\ParkinsonsExp\\%.1fSP\\%s\\%dinputs\\%doutputs\\%dN\\",
                        skipPercentage, hp.name, INPUT_SIZE, OUTPUT_SIZE, N
                    );
                    File baseDirectory = new File(baseDirectoryPath);
                    if (!baseDirectory.exists()) baseDirectory.mkdirs();
                    createCommonLog(baseDirectoryPath, skipPercentage, INPUT_SIZE, OUTPUT_SIZE, N, hp);
                    String[] methods = {"SGD", "Adam", "RMSprop", "DO", "DOAdam", "DORMSprop", "L2", "L2Adam", "L2RMSprop"};
                    for (String method : methods) {
                        totalTasks.incrementAndGet();
                        TrainingConfig config = new TrainingConfig(
                            dataLoader.trainInputs, dataLoader.trainTargets,
                            dataLoader.valInputs, dataLoader.valTargets,
                            INPUT_SIZE, OUTPUT_SIZE, N, skipPercentage, hp, method, baseDirectoryPath
                        );
                        executor.submit(() -> {
                            try {
                                trainMethodThreadSafe(config);
                                int completed = completedTasks.incrementAndGet();
                                System.out.println(String.format(
                                    "[Progress: %d/%d] Completed: %s - %s - N=%d - Skip=%.2f",
                                    completed, totalTasks.get(), config.hp.name, config.methodName, config.N, config.skipPercentage
                                ));
                            } catch (Exception e) {
                                System.err.println("Error in training task: " + e.getMessage());
                                e.printStackTrace();
                            }
                        });
                    }
                }
            }
        }

        executor.shutdown();
        System.out.println("\n========================================");
        System.out.println("All tasks submitted. Waiting for completion...");
        System.out.println("========================================\n");
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            System.err.println("Training interrupted: " + e.getMessage());
        }
        System.out.println("\n========================================");
        System.out.println("ALL TRAINING COMPLETED!");
        System.out.println("Total tasks: " + totalTasks.get());
        System.out.println("========================================");
    }

    private static void createCommonLog(String baseDirectoryPath, double skipPercentage,
                                       int input, int output, int N, HyperparameterSet hp) {
        String commonLogFileName = baseDirectoryPath + "common_log.txt";
        synchronized (DNNSGOP.class) {
            try (PrintWriter commonWriter = new PrintWriter(new FileWriter(commonLogFileName, true))) {
                commonWriter.println("========================================");
                commonWriter.println("ARCHITECTURAL CONFIGURATION:");
                commonWriter.println("  Skip Percentage: " + skipPercentage);
                commonWriter.println("  Input Size: " + input);
                commonWriter.println("  Output Size: " + output);
                commonWriter.println("  N: " + N);
                commonWriter.println("========================================");
                commonWriter.println("HYPERPARAMETER EXPERIMENT: " + hp.name);
                commonWriter.println("  Learning Rate: " + hp.learningRate);
                commonWriter.println("  Dropout Rate: " + hp.dropoutRate);
                commonWriter.println("  L2 Lambda: " + hp.l2Lambda);
                commonWriter.println("  Adam Beta1: " + hp.adamBeta1);
                commonWriter.println("  Adam Beta2: " + hp.adamBeta2);
                commonWriter.println("  RMSprop Decay: " + hp.rmspropDecay);
                commonWriter.println("========================================");
                commonWriter.println("TRAINING CONFIGURATION:");
                commonWriter.println("  Epochs: " + NUM_EPOCHS);
                int[] S = {input, N, N, output};
                network student = new network();
                fnCreate.createNetwork(student, S, skipPercentage, commonWriter);
                int studentLayers = countLayers(student);
                commonWriter.println("Number of layers in Student: " + studentLayers);
                fnCreate.verifyOutputLayerConnections(student, commonWriter);
            } catch (IOException e) {
                System.err.println("Error writing to common log file: " + e.getMessage());
            }
        }
    }

    private static void trainMethodThreadSafe(TrainingConfig config) {
        String methodDirectoryPath = config.baseDirectoryPath + config.methodName + "\\";
        File methodDirectory = new File(methodDirectoryPath);
        synchronized (DNNSGOP.class) {
            if (!methodDirectory.exists()) methodDirectory.mkdirs();
        }
        String logFileName = methodDirectoryPath + "training_log.txt";
        try (PrintWriter writer = new PrintWriter(new FileWriter(logFileName, true))) {
            writer.println("Training with " + config.methodName + " for:");
            writer.println("  Architecture - Skip Percentage: " + config.skipPercentage);
            writer.println("  Hyperparameters: " + config.hp.name);
            writer.println("  Input Size: " + config.input);
            writer.println("  Output Size: " + config.output);
            writer.println("  N: " + config.N);
            int[] S = {config.input, config.N, config.N, config.output};
            network student = new network();
            fnCreate.createNetwork(student, S, config.skipPercentage, writer);
            fnTrain.Tinputs = config.inputs;
            fnTrain.Ttargets = config.targets;
            fnTrain.TinputsValidate = config.inputsValidate;
            fnTrain.TtargetsValidate = config.targetsValidate;
            fnTrain.Student = student;
            fnTrain.Teacher = null;
            switch (config.methodName) {
                case "SGD":
                    fnTrain.Train(NUM_EPOCHS, config.input, config.output, config.N, config.skipPercentage, writer);
                    break;
                case "Adam":
                    fnTrain.TrainAdam(NUM_EPOCHS, config.input, config.output, config.N, config.skipPercentage, writer);
                    break;
                case "RMSprop":
                    fnTrain.TrainRMSprop(NUM_EPOCHS, config.input, config.output, config.N, config.skipPercentage, writer);
                    break;
                case "DO":
                    fnTrain.TrainDO(NUM_EPOCHS, config.input, config.output, config.N, config.skipPercentage, writer);
                    break;
                case "DOAdam":
                    fnTrain.TrainDOAdam(NUM_EPOCHS, config.input, config.output, config.N, config.skipPercentage, writer);
                    break;
                case "DORMSprop":
                    fnTrain.TrainDORMSprop(NUM_EPOCHS, config.input, config.output, config.N, config.skipPercentage, writer);
                    break;
                case "L2":
                    fnTrain.TrainL2(NUM_EPOCHS, config.input, config.output, config.N, config.skipPercentage, writer);
                    break;
                case "L2Adam":
                    fnTrain.TrainL2Adam(NUM_EPOCHS, config.input, config.output, config.N, config.skipPercentage, writer);
                    break;
                case "L2RMSprop":
                    fnTrain.TrainL2RMSprop(NUM_EPOCHS, config.input, config.output, config.N, config.skipPercentage, writer);
                    break;
            }
            writer.println(config.methodName + " training and evaluation completed.");
        } catch (IOException e) {
            System.err.println("Error writing to " + config.methodName + " log file: " + e.getMessage());
        }
    }

    public static int countLayers(network net) {
        int count = 0;
        layernode current = net.inputlayernode;
        while (current != null) {
            count++;
            current = current.next;
        }
        return count;
    }
}