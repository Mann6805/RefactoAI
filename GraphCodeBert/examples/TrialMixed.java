// TrialMixed.java

import java.io.FileReader;
import java.io.IOException;

public class TrialMixed {

    private int counter = 0;

    // --------- Defective methods ---------
    public String readFileUnsafe(String filename) throws IOException {
        FileReader fr = new FileReader(filename);
        // Missing fr.close()
        char[] data = new char[100];
        fr.read(data);
        return new String(data);
    }

    public void processStringUnsafe(String s) {
        if (s == null) {
            System.out.println(s.length()); // NullPointerException
        }
    }

    public void incrementUnsafe() {
        counter++; // Not synchronized
    }

    // --------- Correct methods ---------
    public String readFileSafe(String filename) throws IOException {
        FileReader fr = new FileReader(filename);
        try {
            char[] data = new char[100];
            fr.read(data);
            return new String(data);
        } finally {
            fr.close(); // Properly closed
        }
    }

    public void processStringSafe(String s) {
        if (s == null) {
            System.out.println(""); // Safe
        } else {
            System.out.println(s.length());
        }
    }

    public synchronized void incrementSafe() {
        counter++; // Thread-safe
    }
}
