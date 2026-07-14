#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080

int main() {

    int sock;
    struct sockaddr_in serv_addr;

    int buffer[3] = {5, 3, 2};
    int result[3];

    // Create socket
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket creation failed");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

    // Connect to server
    if (connect(sock, (struct sockaddr *)&serv_addr,
                sizeof(serv_addr)) < 0) {
        perror("Connection failed");
        return -1;
    }

    // Send numbers
    write(sock, buffer, sizeof(buffer));

    // Receive results
    read(sock, result, sizeof(result));

    printf("Numbers Sent : %d %d %d\n",
           buffer[0], buffer[1], buffer[2]);

    printf("Sum        = %d\n", result[0]);
    printf("Difference = %d\n", result[1]);
    printf("Product    = %d\n", result[2]);

    close(sock);

    return 0;
}
// gcc client.c -o client
// ./client