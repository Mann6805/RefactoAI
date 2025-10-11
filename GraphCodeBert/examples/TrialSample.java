// TrialSample.java

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

// 1. Resource leak
public class TrialSample {
    public String readFile(String filename) throws IOException {
        FileReader fr = new FileReader(filename);
        // Missing fr.close()
        char[] data = new char[100];
        fr.read(data);
        return new String(data);
    }

    // 2. Null pointer dereference
    public void processString(String s) {
        if (s == null) {
            System.out.println(s.length()); // Will throw NullPointerException
        }
    }

    // 3. Concurrency issue
    private int counter = 0;
    public void incrementCounter() {
        counter++; // Not synchronized
    }

    // 4. Security vulnerability
    public void runCommand(String cmd) throws IOException {
        Runtime.getRuntime().exec(cmd); // Vulnerable to injection
    }

    // 5. Code complexity
    public void nestedLoops() {
        for(int i=0; i<5; i++) {
            for(int j=0; j<5; j++) {
                for(int k=0; k<5; k++) {
                    System.out.println(i + " " + j + " " + k);
                }
            }
        }
    }
}