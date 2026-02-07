package dnnsg;

import dnnsg.fnTrain.TrainingContext;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class DNNSG {

    // ========== SKIP CONNECTION ANALYSIS PARAMETERS ==========
    static double[] SKIP_PERCENTAGES_TO_TEST = {0.0, 0.25, 0.5, 0.75, 1.00};
    static double CURRENT_SKIP_PERCENTAGE = 0.0;
    
    // Baseline parameter tracking
    static Map<String, Integer> CLASS_BASELINE_PARAMS = new HashMap<>();
    static Map<String, String> BASELINE_STUDENT_CONFIG = new HashMap<>();
    static final double MAX_PARAM_DEVIATION = 0.05;

    // ========== ARCHITECTURE CONFIGURATION ==========
    static int[] inputsize = {21, 15, 9};
    static int[] outputsize = {10, 8, 4};
    static int NUM_REPLICATIONS = 3;

    static int[][] teacherConfigs = {
        {21, 270, 8}, {21, 130, 85, 8}, {21, 80, 80, 80, 8},
    };

    
    static int[][] studentConfigs = {
        {21, 25, 8}, {21, 25, 25, 8}, {21, 25, 25, 25, 8},
    };

    // ========== TRAINING PARAMETERS ==========
    static int numdatapoints = 10000;
    static int numValdatapoints = 1000;
    static int numepochs = 100;

    // ========== MULTITHREADING CONFIGURATION ==========
    // With 32GB RAM, use 3-4x CPU cores for maximum parallelism
    static final int NUM_THREADS = Runtime.getRuntime().availableProcessors() + 2; // 6 threads
    static final long MAX_MEMORY_USAGE = 10 * 1024 * 1024 * 1024L; // 10GB max
    static AtomicInteger completedTasks = new AtomicInteger(0);
    static AtomicInteger totalTasks = new AtomicInteger(0);

    // ========== HYPERPARAMETER CONFIGURATIONS ==========
    static class HyperparameterSet {
        String name;
        String optimizerType; // "ADAM" or "RMSPROP"
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
        network teacher;
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
        int teacherIdx;
        int replication;
        int studentIdx;
        
        TrainingConfig(network teacher, double[][] inputs, double[][] targets,
                      double[][] inputsValidate, double[][] targetsValidate,
                      int input, int output, int N, double skipPercentage,
                      HyperparameterSet hp, String baseDirectoryPath,
                      int[] studentConfig, int teacherIdx, int replication, int studentIdx) {
            this.teacher = deepCopyNetwork(teacher);
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
            this.teacherIdx = teacherIdx;
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

    static String mainTeacherFolder = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\TeacherArchitectures81\\";

    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        try {
            ensureDirectoryExists(mainTeacherFolder + "dummy.txt");

            System.out.println("\n" + "═".repeat(80));
            System.out.println("PHASE 3: HYPERPARAMETER + ARCHITECTURE COMBINED TEST");
            System.out.println("═".repeat(80));
            System.out.println("Testing: How do Hyperparameters + Architectures interact when combined?");
            System.out.println("═".repeat(80));
            System.out.println("Skip Percentages: " + formatSkipArray(SKIP_PERCENTAGES_TO_TEST));
            System.out.println("Student Architectures: 9 per class (matched to baseline params)");
            System.out.println("Hyperparameter Sets: 8 different configurations (4 Adam + 4 RMSprop)");
            System.out.println("Teacher Architectures: 3 per class (Shallow, Medium, Deep)");
            System.out.println("Replications: " + NUM_REPLICATIONS);
            System.out.println("Threads: " + NUM_THREADS);
            System.out.println("═".repeat(80) + "\n");

            // PHASE 1: Calculate baseline parameters
            calculateBaselineParameters();
            
            // PHASE 2: Run multithreaded combined experiments
            runCombinedExperiments();

            System.out.println("\n" + "═".repeat(80));
            System.out.println("✅ PHASE 3 COMPLETE: All Hyperparameter + Architecture combinations tested");
            System.out.println("═".repeat(80));

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            scan.close();
        }
    }

    private static void calculateBaselineParameters() {
        CURRENT_SKIP_PERCENTAGE = 0.0;
        
        System.out.println("\n" + "═".repeat(80));
        System.out.println("PHASE 1: CALCULATING STUDENT BASELINES BY CLASS (0% SKIP)");
        System.out.println("One baseline per (input, output) class - applies to ALL students in that class");
        System.out.println("═".repeat(80) + "\n");
        
        String baselineLogPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\ExperimentSet_Phase3_Combined7\\baseline_params.txt";
        ensureDirectoryExists(baselineLogPath);
        
        try (PrintWriter baselineWriter = new PrintWriter(new FileWriter(baselineLogPath))) {
            baselineWriter.println("════════════════════════════════════════════════════════");
            baselineWriter.println("PHASE 3: SINGLE STUDENT BASELINE PARAMETER PER CLASS");
            baselineWriter.println("First student in each (input, output) class = baseline");
            baselineWriter.println("ALL students and ALL teachers in this class use this baseline");
            baselineWriter.println("════════════════════════════════════════════════════════\n");
            
            for (int input : inputsize) {
                for (int output : outputsize) {
                    List<int[]> matchingStudents = getMatchingStudents(input, output);
                    List<int[]> matchingTeachers = getMatchingTeachers(input, output);
                    
                    if (matchingStudents.isEmpty() || matchingTeachers.isEmpty()) continue;
                    
                    String classKey = getClassKey(input, output);
                    int[] baselineConfig = matchingStudents.get(0);
                    int baselineParams = createAndCountParams(baselineConfig, 0.0);
                    
                    CLASS_BASELINE_PARAMS.put(classKey, baselineParams);
                    BASELINE_STUDENT_CONFIG.put(classKey, configToString(baselineConfig));
                    
                    System.out.println("CLASS: Input=" + input + ", Output=" + output);
                    System.out.println("  Baseline Architecture: " + architectureToString(baselineConfig));
                    System.out.println("  Baseline Parameters: " + baselineParams);
                    System.out.println("  Teachers in this class: " + matchingTeachers.size() + " (all use this baseline)");
                    System.out.println("  Students in this class: " + matchingStudents.size() + " (all match to this baseline)");
                    System.out.println();
                    
                    baselineWriter.println("═══════════════════════════════════════════════");
                    baselineWriter.println("Input=" + input + ", Output=" + output + ":");
                    baselineWriter.println("  Baseline Architecture: " + architectureToString(baselineConfig));
                    baselineWriter.println("  Baseline Parameters: " + baselineParams);
                    baselineWriter.println("  Teachers using this baseline:");
                    for (int i = 0; i < matchingTeachers.size(); i++) {
                        baselineWriter.println("    " + (i+1) + ". " + architectureToString(matchingTeachers.get(i)) + " (" + getTeacherDepth(matchingTeachers.get(i)) + ")");
                    }
                    baselineWriter.println("  Students matching to this baseline:");
                    for (int i = 0; i < matchingStudents.size(); i++) {
                        baselineWriter.println("    " + (i+1) + ". " + architectureToString(matchingStudents.get(i)));
                    }
                    baselineWriter.println();
                }
            }
            
        } catch (IOException e) {
            System.out.println("Error writing baseline file: " + e.getMessage());
        }
        
        System.out.println("✅ Baseline calculation complete - Single baseline per class established\n");
    }

private static void runCombinedExperiments() {
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    
    // 12 Hyperparameter configurations: 4 Adam + 4 RMSprop + 4 L2-only (SGD)
    HyperparameterSet[] hyperparameterSets = {
        // Adam-based configurations (4 variations)
        new HyperparameterSet("HP1_Adam_Baseline", "ADAM", 0.001, 0.0005, 0.99, 0.999, 0.99),
        new HyperparameterSet("HP2_Adam_HighLR", "ADAM", 0.01, 0.0015, 0.99, 0.999, 0.99),
        new HyperparameterSet("HP3_Adam_LowLR", "ADAM", 0.0005, 0.0002, 0.99, 0.999, 0.99),
        new HyperparameterSet("HP4_Adam_HighL2", "ADAM", 0.005, 0.0010, 0.99, 0.999, 0.99),
        
        // RMSprop-based configurations (4 variations)
        new HyperparameterSet("HP5_RMSprop_Baseline", "RMSPROP", 0.001, 0.0005, 0.99, 0.999, 0.99),
        new HyperparameterSet("HP6_RMSprop_HighLR", "RMSPROP", 0.01, 0.0015, 0.99, 0.999, 0.99),
        new HyperparameterSet("HP7_RMSprop_LowDecay", "RMSPROP", 0.0008, 0.0003, 0.99, 0.999, 0.95),
        new HyperparameterSet("HP8_RMSprop_HighDecay", "RMSPROP", 0.003, 0.0009, 0.99, 0.999, 0.999),
        
        // L2 Regularization with SGD (4 variations) - No optimizer, just L2 penalty
        new HyperparameterSet("HP9_L2_Baseline", "L2", 0.001, 0.0005, 0.0, 0.0, 0.0),
        new HyperparameterSet("HP10_L2_HighLR", "L2", 0.01, 0.0015, 0.0, 0.0, 0.0),
        new HyperparameterSet("HP11_L2_LowLR", "L2", 0.0005, 0.0002, 0.0, 0.0, 0.0),
        new HyperparameterSet("HP12_L2_HighL2", "L2", 0.005, 0.0010, 0.0, 0.0, 0.0)
    };

    try {
        // ===== LOOP: Input/Output Classes =====
        for (int input : inputsize) {
            for (int output : outputsize) {
                List<int[]> matchingTeachers = getMatchingTeachers(input, output);
                List<int[]> matchingStudents = getMatchingStudents(input, output);
                
                if (matchingTeachers.isEmpty() || matchingStudents.isEmpty()) continue;
                
                String classKey = getClassKey(input, output);
                Integer classBaseline = CLASS_BASELINE_PARAMS.get(classKey);
                
                System.out.println("\n" + "═".repeat(80));
                System.out.println("TESTING CLASS: Input=" + input + ", Output=" + output);
                System.out.println("Single Baseline=" + classBaseline + " params (applies to all students)");
                System.out.println("═".repeat(80));

                // ===== LOOP: Skip Percentages =====
                for (double skipPercentage : SKIP_PERCENTAGES_TO_TEST) {
                    CURRENT_SKIP_PERCENTAGE = skipPercentage;

                    // ===== LOOP: Teachers (all teachers in class use same baseline) =====
                    for (int teacherIdx = 0; teacherIdx < matchingTeachers.size(); teacherIdx++) {
                        int[] T = matchingTeachers.get(teacherIdx);
                        String teacherDepth = getTeacherDepth(T);

                        // ===== LOOP: Replications =====
                        for (int replication = 0; replication < NUM_REPLICATIONS; replication++) {
                            
                            // Create teacher and datasets (ONCE per replication)
                            network teacher = new network();
                            fnCreate.createNetworkStd(teacher, T);
                            
                            double[][] inputs = new double[numdatapoints][input];
                            double[][] targets = new double[numdatapoints][output];
                            double[][] inputsValidate = new double[numValdatapoints][input];
                            double[][] targetsValidate = new double[numValdatapoints][output];

                            // Generate balanced datasets
                            boolean trainingOk = false;
                            int attempts = 0;
                            while (!trainingOk && attempts++ < 100) {
                                trainingOk = fnTeacher.createDataset(teacher, inputs, targets);
                            }
                            if (!trainingOk) continue;

                            boolean validationOk = false;
                            attempts = 0;
                            while (!validationOk && attempts++ < 100) {
                                validationOk = fnTeacher.createDataset(teacher, inputsValidate, targetsValidate);
                            }
                            if (!validationOk) continue;

                            // ===== LOOP: Students (9 architectures) =====
                            for (int studentIdx = 0; studentIdx < matchingStudents.size(); studentIdx++) {
                                int[] baseStudentConfig = matchingStudents.get(studentIdx);
                                
                                // CRITICAL: Match architecture ONCE per student, BEFORE hyperparameter loop
                                // This ensures Adam, RMSprop, and L2 test the EXACT same student architecture
                                ArchitectureResult archResult = findMatchingArchitecture(baseStudentConfig, classBaseline);
                                
                                if (archResult == null) {
                                    System.err.println("WARNING: Could not match student " + studentIdx + 
                                                     " to baseline " + classBaseline + " for skip " + skipPercentage);
                                    continue;
                                }

                                // ===== LOOP: Hyperparameters (12 sets) =====
                                // ALL HP sets (Adam + RMSprop + L2) use the SAME archResult
                                for (HyperparameterSet hp : hyperparameterSets) {
                                    
                                    String baseDirectoryPath = String.format(
                                        "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\ExperimentSet_Phase3_Combined7\\SkipPercent_%02d\\%dinput_%doutput\\Teacher_%s\\Rep%d\\Student_%d\\HP_%s\\",
                                        (int)(skipPercentage * 100), input, output,
                                        teacherDepth, replication + 1, studentIdx + 1, hp.name
                                    );

                                    totalTasks.incrementAndGet();
                                    
                                    TrainingConfig config = new TrainingConfig(
                                        teacher, inputs, targets, inputsValidate, targetsValidate,
                                        input, output, archResult.config[1], skipPercentage,
                                        hp, baseDirectoryPath, archResult.config,
                                        teacherIdx, replication, studentIdx
                                    );

                                    executor.submit(() -> {
                                        try {
                                            trainCombinedExperiment(config);
                                            int completed = completedTasks.incrementAndGet();
                                            System.out.println(String.format(
                                                "[%d/%d] HP+ARCH: Skip=%.1f%% | Input=%d | Teacher=%s | Student=%d | HP=%s",
                                                completed, totalTasks.get(), config.skipPercentage * 100,
                                                config.input, getTeacherDepth(new int[]{config.input, 100, 100, config.output}),
                                                config.studentIdx + 1, config.hp.name
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
                }
            }
        }
        
        executor.shutdown();
        System.out.println("\n" + "═".repeat(80));
        System.out.println("All tasks submitted. Waiting for " + totalTasks.get() + " training runs to complete...");
        System.out.println("═".repeat(80) + "\n");
        
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
        
        System.out.println("\n" + "═".repeat(80));
        System.out.println("✅ ALL TRAINING COMPLETED!");
        System.out.println("Total training runs: " + totalTasks.get());
        System.out.println("═".repeat(80));
        
    } catch (Exception e) {
        e.printStackTrace();
        executor.shutdownNow();
    }
}

private static void trainCombinedExperiment(TrainingConfig config) {
    ensureDirectoryExists(config.baseDirectoryPath + "dummy.txt");

    String logFileName = config.baseDirectoryPath + "combined_training.txt";
    try (PrintWriter writer = new PrintWriter(new FileWriter(logFileName))) {
        writer.println("════════════════════════════════════════════════════════════════");
        writer.println("PHASE 3: HYPERPARAMETER + ARCHITECTURE COMBINED TEST");
        writer.println("════════════════════════════════════════════════════════════════\n");
        
        writer.println("ARCHITECTURAL CONFIGURATION:");
        writer.println("  Skip Percentage: " + String.format("%.2f%%", config.skipPercentage * 100));
        writer.println("  Input Size: " + config.input);
        writer.println("  Output Size: " + config.output);
        writer.println("  Teacher: " + architectureToString(new int[]{config.input, 100, 100, config.output}));
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
        writer.println("  Teacher Index: " + config.teacherIdx);
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
        writer.println("\nFinal Student Parameters: " + studentParams);
        writer.println("NOTE: This EXACT architecture is tested with ALL 12 HP sets (4 Adam + 4 RMSprop + 4 L2)");
        writer.println("════════════════════════════════════════════════════════════════\n");

        // Create training context
        TrainingContext context = new TrainingContext(
            config.inputs, config.targets, config.inputsValidate, config.targetsValidate,
            config.teacher, student
        );

        // Train with the appropriate optimizer based on configuration
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
        System.err.println("Error in combined experiment: " + e.getMessage());
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
            String tempLog = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\temp_param_check.txt";
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

    private static List<int[]> getMatchingTeachers(int input, int output) {
        List<int[]> matching = new ArrayList<>();
        for (int[] config : teacherConfigs) {
            if (config[0] == input && config[config.length - 1] == output) {
                matching.add(config);
            }
        }
        return matching;
    }

    private static List<int[]> getMatchingStudents(int input, int output) {
        List<int[]> matching = new ArrayList<>();
        for (int[] config : studentConfigs) {
            if (config[0] == input && config[config.length - 1] == output) {
                matching.add(config);
            }
        }
        return matching;
    }

    private static String getTeacherDepth(int[] config) {
        int hiddenLayers = config.length - 2;
        if (hiddenLayers == 1) return "Shallow";
        if (hiddenLayers == 2) return "Medium";
        if (hiddenLayers == 3) return "Deep";
        return "Custom";
    }

    private static String getConfigLabel(int[] config) {
        StringBuilder label = new StringBuilder();
        for (int i = 0; i < config.length; i++) {
            label.append(config[i]);
            if (i == 0) label.append("n");
            else if (i == config.length - 1) label.append("o");
            else label.append("n");
            if (i < config.length - 1) label.append("-");
        }
        return label.toString();
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

    private static String calculateTotalExperiments() {
        // skipPercentages × inputs × outputs × teachers × replications × students × hyperparameters
        int total = SKIP_PERCENTAGES_TO_TEST.length * inputsize.length * outputsize.length * 3 * NUM_REPLICATIONS * 9 * 8;
        double hoursPerExperiment = 8.3 * 24 / total;  // 8.3 days / total experiments = hours per experiment
        double estimatedHours = (total / (double) NUM_THREADS) * hoursPerExperiment;
        double estimatedDays = estimatedHours / 24;
        return String.format("%,d experiments | Est. %.2f days with %d threads", total, estimatedDays, NUM_THREADS);
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

    private static network deepCopyNetwork(network original) {
        // TODO: Implement proper deep copy based on your network structure
        return original;
    }
}