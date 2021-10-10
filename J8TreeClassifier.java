package com.WekaUi;

import java.io.File;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class J8TreeClassifier {

    private final static String CONFIDENCE_PRUNING_THRESHOLD = "-C";
    private final static String CONFIDENCE_PRUNING_THRESHOLD_VALUE = "0.25";
    private final static String MIN_LEAF_INSTANCES = "-M";
    private final static String MIN_LEAF_INSTANCES_VALUE = "2";

    private final static String[] J48_OPTIONS = {CONFIDENCE_PRUNING_THRESHOLD, CONFIDENCE_PRUNING_THRESHOLD_VALUE,
        MIN_LEAF_INSTANCES, MIN_LEAF_INSTANCES_VALUE};

    /**
     *
     */
    public String evaluate(String trainingFileInput, String testFileInput) throws Exception {
        StringBuilder response = new StringBuilder();
        String trainFilePath = trainingFileInput;
        String testFilePath = testFileInput;

        File trainFile = getTrainFile(trainFilePath);
        File testFile = getTestFile(testFilePath);

        DataSourceUtils dataSourceUtils = new DataSourceUtils();
        dataSourceUtils.newWekaInstances(trainFile);

        System.out.println("Cargadas las instancias del fichero de entrenamiento y testeo.");

        response.append("Cargadas las instancias del fichero de entrenamiento y testeo.\n");

        ArffLoader loader = new ArffLoader();
        loader.setFile(trainFile);
        Instances trainInstances = loader.getStructure();
        trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

        Instances testInstances = dataSourceUtils.newWekaInstances(testFile);

        J48 treeClassifier = new J48();
        treeClassifier.setOptions(J48_OPTIONS);
        treeClassifier.buildClassifier(trainInstances);

        Evaluation evaluation = new Evaluation(testInstances);

        evaluation.evaluateModel(treeClassifier, testInstances);

        boolean printComplexityStatistics = false;

        System.out.println(evaluation.toSummaryString(printComplexityStatistics));

        System.out.println(evaluation.toMatrixString());

        response.append(evaluation.toSummaryString(printComplexityStatistics));

        response.append(evaluation.toMatrixString());

        return response.toString();
    }

    /**
     *
     */
    private static File getTrainFile(String trainFilePath) throws Exception {

        File trainFile = new File(trainFilePath);

        if (!trainFile.exists()) {
            System.out.println("No existe el fichero de entrenamiento [" + trainFilePath + "]. "
                    + "Se autogenerará uno con 10 millones de instancias.");
            trainFile = DataSourceUtils.createLed24DataSource(trainFilePath, 10000000);
        }

        return trainFile;
    }

    /**
     *
     */
    private static File getTestFile(String testFilePath) throws Exception {

        File testFile = new File(testFilePath);

        if (!testFile.exists()) {
            System.out.println("No existe el fichero de testeo [" + testFilePath + "]. "
                    + "Se autogenerará uno con 10 mil instancias.");
            testFile = DataSourceUtils.createLed24DataSource(testFilePath, 10000);
        }

        return testFile;
    }

}
