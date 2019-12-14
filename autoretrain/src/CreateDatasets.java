//
// Copyright © 2019 Attila Ulbert
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
// files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, 
// modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
// is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;


public class CreateDatasets {

    static final String DEFAULT_JDBC_DRIVER = "org.postgresql.Driver";
    static final String DB_URL = "jdbc:postgresql://localhost:5432/dhmedrecords";

    static final String LABELS_FILE_PATH = "models/labels.json";

//    static final String DEFAULT_JDBC_DRIVER = "org.hsqldb.jdbc.JDBCDriver";
//    static final String DB_URL = "jdbc:hsqldb:file:\\c:\\Users\\eattulb\\work\\dh\\dhb\\db\\dhmedrecords.db;hsqldb.lock_file=false";

    static final String USER = "dhmedrecords";
    static final String PASS = "dhmedrecords";

    static final SimpleDateFormat dateFormatter = new SimpleDateFormat("dd-MM-yyyy_HH:mm:ss");

    private java.util.Date trainFromTime;
    private java.util.Date trainToTime;
    private java.util.Date testFromTime;
    private java.util.Date testToTime;

    private Connection conn = null;
    private String datasetRoot = "";
    private String patientRoot;
    private String recordRoot;

    private Set<Integer> testMedrecIds = new HashSet<Integer>();

    public static void main(String[] args) {
        try {
            CreateDatasets exporter = new CreateDatasets(args);

            exporter.run();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void run() {
        try {
            System.out.println("Creating subdirs.");
            createSubdirs();

            System.out.println("Creating symlinks to baseline test images.");
            Set<Integer> testSetMedrecIds = symlinksToBaselineTestImages();

            System.out.println("Exporting test images.");
            testSetMedrecIds.addAll(exportImages("test", testFromTime, testToTime, testSetMedrecIds));

            System.out.println("Exporting training images.");
            exportImages("train", trainFromTime, trainToTime, testSetMedrecIds);
        } catch (SQLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (org.json.simple.parser.ParseException e) {
            e.printStackTrace();
        } finally {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    private CreateDatasets(String[] args) throws ClassNotFoundException, SQLException, ParseException, IOException, org.json.simple.parser.ParseException {
        datasetRoot = args[0];

        trainFromTime = dateFormatter.parse(args[1]);
        trainToTime = dateFormatter.parse(args[2]);

        testFromTime = dateFormatter.parse(args[3]);
        testToTime = dateFormatter.parse(args[4]);

        String jdbcDriver = System.getProperty("jdbc.driver");
        Class.forName(jdbcDriver != null ? jdbcDriver : DEFAULT_JDBC_DRIVER);

        conn = DriverManager.getConnection(DB_URL, USER, PASS);
        conn.setAutoCommit(false);
    }

    private void createSubdirs() throws IOException, org.json.simple.parser.ParseException {
        JSONArray labelsJson = (JSONArray) new JSONParser().parse(new FileReader(LABELS_FILE_PATH));

        Iterator<String> iterator = labelsJson.iterator();
        while (iterator.hasNext()) {
            final String categoryName = iterator.next();
            System.out.println(categoryName);

            new File(datasetRoot + File.separator + "train" + File.separator + categoryName).mkdir();
            new File(datasetRoot + File.separator + "test" + File.separator + categoryName).mkdir();
        }
    }

    private Set<Integer> symlinksToBaselineTestImages() {
        HashSet<Integer> medrecIds = new HashSet<Integer>();

        File[] dirs = new File(datasetRoot + File.separator + "BASELINE_TEST").listFiles();
        for (File dir : dirs) {
            System.out.println(dir.getName());

            File[] files = dir.listFiles();
            for (File file : files) {
                Path targetPath = null;
                try {
                    targetPath = Paths.get(
                            new File(datasetRoot).getAbsolutePath(), "test", dir.getName(), file.getName());
                    System.out.println(file.toPath() + " -> " + targetPath);

                    Files.createSymbolicLink(targetPath, Paths.get(file.getAbsolutePath()));
                } catch (IOException e) {
                    System.err.println("Link cannot be created to " + file.getName() + " from " + targetPath);
                }

                int medrecId = Integer.parseInt(file.getName().split("-")[1]);
                medrecIds.add(medrecId);
            }
        }

        return medrecIds;
    }

    private String getDiagCode(String diag) {
        String diagCode = null;

        if (diag != null && diag.length() >= 4) {
            diagCode = diag.substring(diag.indexOf('#') + 1);

            int endIndex = diagCode.indexOf(' ');
            if (endIndex >= 1) {
                diagCode = diagCode.substring(0, endIndex);
            } else {
                diagCode = null;
            }
        }

        return diagCode;
    }

    private Set<Integer> exportImages(String kind, java.util.Date fromTime, java.util.Date toTime, Set<Integer> excludeMedrecIds) throws SQLException, IOException {
        HashSet<Integer> medrecIds = new HashSet<Integer>();

        PreparedStatement stmt = conn.prepareStatement
                ("SELECT b.id, b.data, m.diagnosis, m.id, d.creationtime FROM Binarycontent b, Document d, Medicalrecord_Document md, Medicalrecord m " +
                        "WHERE d.type='skinMicroPic' AND m.diagnosis<>'' AND m.id=md.medicalrecord_id AND md.document_id=d.id AND d.content_id=b.id AND " +
                        "d.creationtime>? AND ?>=d.creationtime");

        stmt.setTimestamp(1, new Timestamp(fromTime.getTime()));
        stmt.setTimestamp(2, new Timestamp(toTime.getTime()));

        ResultSet rs = stmt.executeQuery();
        while (rs.next()) {
            int id = rs.getInt(1);
            Blob image = rs.getBlob(2);
            String diagnosis = rs.getString(3);
            int medrecId = rs.getInt(4);
            Timestamp creationTime = rs.getTimestamp(5);

            System.out.print("  TS: " + creationTime.toString());
            System.out.print(" MID: " + medrecId);
            System.out.print("  ID: " + id);
            System.out.print("   D: " + diagnosis);

            medrecIds.add(medrecId);
            if (excludeMedrecIds.contains(medrecId)) {
                System.out.println(" Image " + id + " of medrec " + medrecId + " is already used. Skipping image.");
                continue;
            }

            final String diagCode = getDiagCode(diagnosis);
            System.out.print(" DC: " + diagCode);
            if (diagCode != null) {
                final String targetDirPath = datasetRoot + File.separator + kind + File.separator + diagCode;
                if (new File(targetDirPath).exists()) {
                    InputStream is = image.getBinaryStream();
                    FileOutputStream fos = new FileOutputStream(
                            targetDirPath + File.separator + medrecId + "-" + id + ".jpg");
                    try {
                        fos.write(is.readAllBytes());
                    } finally {
                        fos.close();
                        is.close();
                    }
                } else {
                    System.out.println("Category dir does not exist for diag code!");
                }
            } else {
                System.out.println("Empty diag code!");
            }
            System.out.println();
        }
        rs.close();
        stmt.close();

        return medrecIds;
    }
}
