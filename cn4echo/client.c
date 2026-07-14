#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUF_SIZE 1024

int main()
{
    int sock;
    struct sockaddr_in serv_addr;

    char message[BUF_SIZE];
    char buffer[BUF_SIZE] = {0};

    // Create socket
    sock = socket(AF_INET, SOCK_STREAM, 0);

    if (sock < 0)
    {
        perror("Socket creation failed");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

    // Connect
    if (connect(sock,
                (struct sockaddr *)&serv_addr,
                sizeof(serv_addr)) < 0)
    {
        perror("Connection failed");
        return -1;
    }

    printf("Enter message: ");
    fgets(message, BUF_SIZE, stdin);

    // Send
    send(sock, message, strlen(message), 0);

    // Receive echo
    int valread = read(sock, buffer, BUF_SIZE);

    if (valread > 0)
    {
        buffer[valread] = '\0';
        printf("Echo from server: %s", buffer);
    }

    close(sock);

    return 0;
}

// gnn client.c -o client
// ./client