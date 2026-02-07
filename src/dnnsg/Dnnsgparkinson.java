package dnnsg;

import dnnsg.fnTrain.TrainingContext;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * DNNSG PARKINSON'S EXPERIMENT
 * 
 * Mirrors DNNSG.java structure but for REAL Parkinson's medical data
 * KEY DIFFERENCE: STUDENT-ONLY training (NO teachers, unlike DNNSG)
 * 
 * Tests skip connections on real-world Parkinson's diagnosis data
 * Validates DNNSG's findings on actual medical classification task
 * 
 * @author Your Name
 */
public class Dnnsgparkinson {

    // ========== SKIP CONNECTION ANALYSIS PARAMETERS ==========
    static double[] SKIP_PERCENTAGES_TO_TEST = {0.0, 0.25, 0.5, 0.75, 1.00};
    static double CURRENT_SKIP_PERCENTAGE = 0.0;
    
    // Baseline parameter tracking
    static Map<String, Integer> CLASS_BASELINE_PARAMS = new HashMap<>();
    static Map<String, String> BASELINE_STUDENT_CONFIG = new HashMap<>();
    static final double MAX_PARAM_DEVIATION = 0.05;

    // ========== ARCHITECTURE CONFIGURATION ==========
    // Parkinson's: Fixed 22 inputs, 1 output (binary classification)
    static int inputsize = 22;
    static int outputsize = 1;
    static int NUM_REPLICATIONS = 3;

    // Student configurations (9 total) - progressive depth
    static int[][] studentConfigs = {
        {22, 16, 1},                                          // 0: Shallow (3 layers)
        {22, 12, 12, 1},                                      // 1: Medium (4 layers)
        {22, 10, 10, 10, 1},                                  // 2: Medium-Deep (5 layers)
        {22, 8, 8, 8, 8, 1},                                  // 3: Deep (6 layers)
        {22, 6, 6, 6, 6, 6, 1},                               // 4: Deeper (7 layers)
        {22, 5, 5, 5, 5, 5, 5, 1},                            // 5: Very Deep (8 layers)
        {22, 4, 4, 4, 4, 4, 4, 4, 4, 1},                      // 6: Very Deep (10 layers)
        {22, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1},                // 7: Extremely Deep (12 layers)
        {22, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1}        // 8: Ultra Deep (15 layers)
    };

    // ========== TRAINING PARAMETERS ==========
    static int numepochs = 100;

    // ========== MULTITHREADING CONFIGURATION ==========
    static final int NUM_THREADS = Runtime.getRuntime().availableProcessors() + 2;
    static final long MAX_MEMORY_USAGE = 10 * 1024 * 1024 * 1024L; // 10GB max
    static AtomicInteger completedTasks = new AtomicInteger(0);
    static AtomicInteger totalTasks = new AtomicInteger(0);

    // ========== HYPERPARAMETER CONFIGURATIONS ==========
    static class HyperparameterSet {
        String name;
        String optimizerType; // "ADAM", "RMSPROP", or "L2"
        double learningRate;
        double l2Lambda;
        double adamBeta1;
        double adamBeta2;
        double rmspropDecay;
        
        HyperparameterSet(String name, String optimizer, double lr, double l2,
                         double beta1, double beta2, double decay) {
            this.name = name;
            this.optimizerType = optimizer;
            this.learningRate = lr;
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
        String baseDirectoryPath;
        int[] studentConfig;
        int replication;
        int studentIdx;
        
        TrainingConfig(double[][] inputs, double[][] targets,
                      double[][] inputsValidate, double[][] targetsValidate,
                      int input, int output, int N, double skipPercentage,
                      HyperparameterSet hp, String baseDirectoryPath,
                      int[] studentConfig, int replication, int studentIdx) {
            this.inputs = inputs;
            this.targets = targets;
            this.inputsValidate = inputsValidate;
            this.targetsValidate = targetsValidate;
            this.input = input;
            this.output = output;
            this.N = N;
            this.skipPercentage = skipPercentage;
            this.hp = hp;
            this.baseDirectoryPath = baseDirectoryPath;
            this.studentConfig = studentConfig;
            this.replication = replication;
            this.studentIdx = studentIdx;
        }
    }

    static class ArchitectureResult {
        int[] config;
        double skipPercentage;
        int actualParams;

        ArchitectureResult(int[] config, double skipPercentage, int actualParams) {
            this.config = config;
            this.skipPercentage = skipPercentage;
            this.actualParams = actualParams;
        }
    }

    static String mainOutputFolder = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\MyExP3\\Parkinsons\\";
    static String DATASET_PATH = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\Real Data\\Datasets\\data\\processed\\parkinsons\\parkinsons.csv";

    public static void main(String[] args) {
        try {
            ensureDirectoryExists(mainOutputFolder + "dummy.txt");

            System.out.println("\n" + "═".repeat(80));
            System.out.println("DNNSG PARKINSON'S EXPERIMENT: SKIP CONNECTIONS ON REAL MEDICAL DATA");
            System.out.println("═".repeat(80));
            System.out.println("Dataset: Real Parkinson's Disease Medical Data (196 samples, 22 features)");
            System.out.println("Training: STUDENT-ONLY (NO teachers, unlike DNNSG)");
            System.out.println("Task: Binary classification (0=No PD, 1=Has PD)");
            System.out.println("═".repeat(80));
            System.out.println("Skip Percentages: " + formatSkipArray(SKIP_PERCENTAGES_TO_TEST));
            System.out.println("Student Architectures: " + studentConfigs.length + " (matched to baseline)");
            System.out.println("Hyperparameter Sets: 12 (4 Adam + 4 RMSprop + 4 L2)");
            System.out.println("Replications: " + NUM_REPLICATIONS);
            System.out.println("Threads: " + NUM_THREADS);
            System.out.println("═".repeat(80) + "\n");

            // Load Parkinson's dataset
            System.out.println("Loading Parkinson's dataset...");
            ParkinsonsDataLoader loader = new ParkinsonsDataLoader();
            loader.load(DATASET_PATH, inputsize, outputsize);
            System.out.println("✓ Dataset loaded: " + loader.trainInputs.length + " training, " + 
                             loader.valInputs.length + " validation\n");

            // PHASE 1: Calculate baseline parameters
            calculateBaselineParameters();
            
            // PHASE 2: Run multithreaded combined experiments
            runCombinedExperiments(loader);

            System.out.println("\n" + "═".repeat(80));
            System.out.println("✅ DNNSG PARKINSON'S COMPLETE!");
            System.out.println("Dataset: Real Parkinson's medical data");
            System.out.println("Results saved to: " + mainOutputFolder);
            System.out.println("═".repeat(80) + "\n");
            
        } catch (Exception e) {
            System.err.println("❌ ERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void calculateBaselineParameters() {
        CURRENT_SKIP_PERCENTAGE = 0.0;
        
        System.out.println("\n" + "═".repeat(80));
        System.out.println("PHASE 1: CALCULATING STUDENT BASELINE PARAMETERS (0% SKIP)");
        System.out.println("One baseline for all students: Input=" + inputsize + ", Output=" + outputsize);
        System.out.println("═".repeat(80) + "\n");
        
        String baselineLogPath = mainOutputFolder + "baseline_params.txt";
        ensureDirectoryExists(baselineLogPath);
        
        try (PrintWriter baselineWriter = new PrintWriter(new FileWriter(baselineLogPath))) {
            baselineWriter.println("════════════════════════════════════════════════════════════════");
            baselineWriter.println("DNNSG PARKINSON'S: STUDENT BASELINE PARAMETERS (0% SKIP)");
            baselineWriter.println("Dataset: Real Parkinson's Disease Medical Data");
            baselineWriter.println("Training: Student-only (NO teachers)");
            baselineWriter.println("════════════════════════════════════════════════════════════════\n");
            
            String classKey = getClassKey(inputsize, outputsize);
            
            // Use first student as baseline
            int[] baselineConfig = studentConfigs[0];
            int baselineParams = createAndCountParams(baselineConfig, 0.0);
            
            CLASS_BASELINE_PARAMS.put(classKey, baselineParams);
            BASELINE_STUDENT_CONFIG.put(classKey, configToString(baselineConfig));
            
            System.out.println("CLASS: Input=" + inputsize + ", Output=" + outputsize);
            System.out.println("  ✓ Baseline Architecture: " + architectureToString(baselineConfig));
            System.out.println("  ✓ Baseline Parameters: " + baselineParams);
            System.out.println("  ✓ All " + studentConfigs.length + " student architectures will match this baseline ±5%");
            System.out.println();
            
            baselineWriter.println("CLASS: Input=" + inputsize + ", Output=" + outputsize);
            baselineWriter.println("  Baseline Architecture: " + architectureToString(baselineConfig));
            baselineWriter.println("  Baseline Parameters: " + baselineParams);
            baselineWriter.println("  Students (all matched to baseline ±5%):");
            for (int i = 0; i < studentConfigs.length; i++) {
                int params = createAndCountParams(studentConfigs[i], 0.0);
                double deviation = 100.0 * Math.abs(params - baselineParams) / baselineParams;
                baselineWriter.println("    " + (i+1) + ". " + architectureToString(studentConfigs[i]) + 
                                     " (" + params + " params, " + String.format("%.2f%%", deviation) + " dev)");
            }
            baselineWriter.println("\n════════════════════════════════════════════════════════════════");
            baselineWriter.println("All students tested with 12 hyperparameter sets and 5 skip percentages");
            baselineWriter.println("════════════════════════════════════════════════════════════════");
            
        } catch (IOException e) {
            System.out.println("Error writing baseline file: " + e.getMessage());
        }
        
        System.out.println("✅ Baseline calculation complete\n");
    }

    private static void runCombinedExperiments(ParkinsonsDataLoader loader) {
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        
        // 12 Hyperparameter configurations
        HyperparameterSet[] hyperparameterSets = {
            // ADAM (4)
            new HyperparameterSet("HP1_Adam_Baseline", "ADAM", 0.001, 0.0005, 0.99, 0.999, 0.99),
            new HyperparameterSet("HP2_Adam_HighLR", "ADAM", 0.01, 0.0015, 0.99, 0.999, 0.99),
            new HyperparameterSet("HP3_Adam_LowLR", "ADAM", 0.0005, 0.0002, 0.99, 0.999, 0.99),
            new HyperparameterSet("HP4_Adam_HighL2", "ADAM", 0.005, 0.0010, 0.99, 0.999, 0.99),
            
            // RMSPROP (4)
            new HyperparameterSet("HP5_RMSprop_Baseline", "RMSPROP", 0.001, 0.0005, 0.99, 0.999, 0.99),
            new HyperparameterSet("HP6_RMSprop_HighLR", "RMSPROP", 0.01, 0.0015, 0.99, 0.999, 0.99),
            new HyperparameterSet("HP7_RMSprop_LowDecay", "RMSPROP", 0.0008, 0.0003, 0.99, 0.999, 0.95),
            new HyperparameterSet("HP8_RMSprop_HighDecay", "RMSPROP", 0.003, 0.0009, 0.99, 0.999, 0.999),
            
            // L2 (4)
            new HyperparameterSet("HP9_L2_Baseline", "L2", 0.001, 0.0005, 0.0, 0.0, 0.0),
            new HyperparameterSet("HP10_L2_HighLR", "L2", 0.01, 0.0015, 0.0, 0.0, 0.0),
            new HyperparameterSet("HP11_L2_LowLR", "L2", 0.0005, 0.0002, 0.0, 0.0, 0.0),
            new HyperparameterSet("HP12_L2_HighL2", "L2", 0.005, 0.0010, 0.0, 0.0, 0.0)
        };

        try {
            String classKey = getClassKey(inputsize, outputsize);
            Integer classBaseline = CLASS_BASELINE_PARAMS.get(classKey);
            
            System.out.println("\n" + "═".repeat(80));
            System.out.println("PHASE 2: RUNNING COMBINED EXPERIMENTS");
            System.out.println("Baseline=" + classBaseline + " params (applies to all students)");
            System.out.println("═".repeat(80));

            // ===== LOOP: Skip Percentages =====
            for (double skipPercentage : SKIP_PERCENTAGES_TO_TEST) {
                CURRENT_SKIP_PERCENTAGE = skipPercentage;
                
                System.out.println("\n" + "─".repeat(80));
                System.out.println("SKIP CONNECTIONS: " + String.format("%.1f%%", skipPercentage * 100));
                System.out.println("─".repeat(80));

                // ===== LOOP: Replications =====
                for (int replication = 0; replication < NUM_REPLICATIONS; replication++) {
                    
                    // ===== LOOP: Students (9 architectures) =====
                    for (int studentIdx = 0; studentIdx < studentConfigs.length; studentIdx++) {
                        int[] baseStudentConfig = studentConfigs[studentIdx];
                        
                        // Match architecture
                        ArchitectureResult archResult = findMatchingArchitecture(baseStudentConfig, classBaseline);
                        
                        if (archResult == null) {
                            System.err.println("WARNING: Could not match student " + studentIdx + 
                                             " to baseline " + classBaseline + " for skip " + skipPercentage);
                            continue;
                        }

                        // ===== LOOP: Hyperparameters (12 sets) =====
                        for (HyperparameterSet hp : hyperparameterSets) {
                            
                            String baseDirectoryPath = String.format(
                                "%sSkipPercent_%02d\\Rep%d\\Student_%d\\HP_%s\\",
                                mainOutputFolder,
                                (int)(skipPercentage * 100), replication + 1, studentIdx + 1, hp.name
                            );

                            totalTasks.incrementAndGet();
                            
                            TrainingConfig config = new TrainingConfig(
                                loader.trainInputs, loader.trainTargets,
                                loader.valInputs, loader.valTargets,
                                inputsize, outputsize, loader.trainInputs.length, skipPercentage,
                                hp, baseDirectoryPath, archResult.config,
                                replication, studentIdx
                            );

                            executor.submit(() -> {
                                try {
                                    trainCombinedExperiment(config);
                                    int completed = completedTasks.incrementAndGet();
                                    System.out.println(String.format(
                                        "[%d/%d] Skip=%.1f%% | Rep=%d | Student=%d | HP=%s",
                                        completed, totalTasks.get(), config.skipPercentage * 100,
                                        config.replication + 1, config.studentIdx + 1, config.hp.name
                                    ));
                                } catch (Exception e) {
                                    System.err.println("Error: " + e.getMessage());
                                    e.printStackTrace();
                                }
                            });
                        }
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            executor.shutdownNow();
        }
        
        executor.shutdown();
        System.out.println("\n" + "═".repeat(80));
        System.out.println("All " + totalTasks.get() + " tasks submitted. Waiting for completion...");
        System.out.println("═".repeat(80));
        
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            e.printStackTrace();
            executor.shutdownNow();
        }
        
        System.out.println("\n" + "═".repeat(80));
        System.out.println("✅ ALL TRAINING COMPLETED! Total runs: " + totalTasks.get());
        System.out.println("═".repeat(80));
    }

    private static void trainCombinedExperiment(TrainingConfig config) {
        ensureDirectoryExists(config.baseDirectoryPath + "dummy.txt");

        String logFileName = config.baseDirectoryPath + "training_log.txt";
        try (PrintWriter writer = new PrintWriter(new FileWriter(logFileName))) {
            writer.println("════════════════════════════════════════════════════════════════");
            writer.println("DNNSG PARKINSON'S: STUDENT-ONLY TRAINING");
            writer.println("════════════════════════════════════════════════════════════════\n");
            
            writer.println("DATASET INFORMATION:");
            writer.println("  Type: REAL Parkinson's Disease Medical Data");
            writer.println("  Training samples: " + config.inputs.length);
            writer.println("  Validation samples: " + config.inputsValidate.length);
            writer.println("  Input features: 22");
            writer.println("  Output: Binary classification (0=No PD, 1=Has PD)");
            writer.println();
            
            writer.println("ARCHITECTURAL CONFIGURATION:");
            writer.println("  Skip Percentage: " + String.format("%.2f%%", config.skipPercentage * 100));
            writer.println("  Input Size: " + config.input);
            writer.println("  Output Size: " + config.output);
            writer.println("  Teacher: NONE (Student-only training)");
            writer.println("  Student: " + architectureToString(config.studentConfig));
            writer.println();
            
            writer.println("HYPERPARAMETER CONFIGURATION:");
            writer.println("  Name: " + config.hp.name);
            writer.println("  Optimizer: " + config.hp.optimizerType);
            writer.println("  Learning Rate: " + config.hp.learningRate);
            writer.println("  L2 Lambda: " + config.hp.l2Lambda);
            if (config.hp.optimizerType.equals("ADAM")) {
                writer.println("  Adam Beta1: " + config.hp.adamBeta1);
                writer.println("  Adam Beta2: " + config.hp.adamBeta2);
            } else if (config.hp.optimizerType.equals("RMSPROP")) {
                writer.println("  RMSprop Decay: " + config.hp.rmspropDecay);
            }
            writer.println();
            
            writer.println("REPLICATION & INDICES:");
            writer.println("  Replication: " + config.replication);
            writer.println("  Student Index: " + config.studentIdx);
            writer.println();
            
            writer.println("PARAMETER MATCHING:");
            writer.println("  Class Baseline: " + CLASS_BASELINE_PARAMS.get(getClassKey(config.input, config.output)));
            writer.println("  Student Config Used: " + architectureToString(config.studentConfig));
            writer.println();

            // Create student network
            int[] S = config.studentConfig;
            network student = new network();
            fnCreate.createNetwork(student, S, config.skipPercentage, writer);
            
            int studentParams = countNetworkParameters(student);
            writer.println("Final Student Parameters: " + studentParams);
            writer.println("════════════════════════════════════════════════════════════════\n");

            // Create training context (NO TEACHER)
            TrainingContext context = new TrainingContext(
                config.inputs, config.targets, config.inputsValidate, config.targetsValidate,
                null,      // NO TEACHER
                student    // STUDENT ONLY
            );

            // Train with appropriate optimizer
            if (config.hp.optimizerType.equals("ADAM")) {
                writer.println("Training with: ADAM OPTIMIZER + L2 REGULARIZATION\n");
                fnTrain.TrainL2Adam(context, numepochs, config.input, config.output, config.N, 
                                   config.skipPercentage, writer, 
                                   config.hp.learningRate, config.hp.l2Lambda,
                                   config.hp.adamBeta1, config.hp.adamBeta2);
            } 
            else if (config.hp.optimizerType.equals("RMSPROP")) {
                writer.println("Training with: RMSPROP OPTIMIZER + L2 REGULARIZATION\n");
                fnTrain.TrainL2RMSprop(context, numepochs, config.input, config.output, config.N, 
                                      config.skipPercentage, writer,
                                      config.hp.learningRate, config.hp.l2Lambda, 
                                      config.hp.rmspropDecay);
            }
            else if (config.hp.optimizerType.equals("L2")) {
                writer.println("Training with: STOCHASTIC GRADIENT DESCENT + L2 REGULARIZATION\n");
                fnTrain.TrainL2(context, numepochs, config.input, config.output, config.N, 
                               config.skipPercentage, writer,
                               config.hp.learningRate, config.hp.l2Lambda);
            }
            else {
                throw new IllegalArgumentException("Unknown optimizer type: " + config.hp.optimizerType);
            }

            writer.println("\n════════════════════════════════════════════════════════════════");
            writer.println("Training completed successfully");
            writer.println("════════════════════════════════════════════════════════════════");

        } catch (IOException e) {
            System.err.println("Error in training: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // ========== HELPER METHODS ==========
    
    private static ArchitectureResult findMatchingArchitecture(int[] baseConfig, int baseline) {
        final int MAX_ITERATIONS = 100;
        int minParams = (int) Math.floor(baseline * (1 - MAX_PARAM_DEVIATION));
        int maxParams = (int) Math.ceil(baseline * (1 + MAX_PARAM_DEVIATION));

        int baseParams = createAndCountParams(baseConfig, CURRENT_SKIP_PERCENTAGE);
        if (baseParams >= minParams && baseParams <= maxParams) {
            return new ArchitectureResult(baseConfig, CURRENT_SKIP_PERCENTAGE, baseParams);
        }

        int[] current = baseConfig.clone();
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            int currentParams = createAndCountParams(current, CURRENT_SKIP_PERCENTAGE);
            
            if (currentParams >= minParams && currentParams <= maxParams) {
                return new ArchitectureResult(current, CURRENT_SKIP_PERCENTAGE, currentParams);
            }
            
            double targetRatio = (double) baseline / currentParams;
            if (current.length > 2) {
                for (int i = 1; i < current.length - 1; i++) {
                    current[i] = Math.max(2, (int) Math.round(current[i] * targetRatio));
                }
            }
        }
        
        return null;
    }

    private static int createAndCountParams(int[] config, double skipPercent) {
        try {
            network testNet = new network();
            String tempLog = mainOutputFolder + "temp_param_check.txt";
            ensureDirectoryExists(tempLog);
            
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempLog))) {
                fnCreate.createNetwork(testNet, config, skipPercent, writer);
                return countNetworkParameters(testNet);
            }
        } catch (Exception e) {
            return -1;
        }
    }

    private static void ensureDirectoryExists(String filePath) {
        File file = new File(filePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
    }

    private static String getClassKey(int input, int output) {
        return input + "_" + output;
    }

    private static String configToString(int[] config) {
        return Arrays.toString(config);
    }

    private static String architectureToString(int[] config) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < config.length; i++) {
            sb.append(config[i]);
            if (i == 0) sb.append("n");
            else if (i == config.length - 1) sb.append("o");
            else sb.append("n");
            if (i < config.length - 1) sb.append("-");
        }
        return sb.toString();
    }

    private static String formatSkipArray(double[] skips) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < skips.length; i++) {
            sb.append(String.format("%.0f%%", skips[i] * 100));
            if (i < skips.length - 1) sb.append(", ");
        }
        return sb.toString();
    }

    public static int countNetworkParameters(network net) {
        int totalParams = 0;
        layernode layer = net.inputlayernode;

        while (layer != null) {
            node n = layer.firstnode;
            while (n != null) {
                if (layer != net.inputlayernode) {
                    totalParams++;
                }
                edge e = n.firstedge;
                while (e != null) {
                    totalParams++;
                    e = e.next;
                }
                n = n.next;
            }
            layer = layer.next;
        }
        return totalParams;
    }
}