---
title: Least Privilege
description: Ausstattung von Nutzern und Prozessen nur mit den minimal
  notwendigen Rechten.
category:
- Security
problems:
- authorization-flaws
- data-protection-risk
- authentication-bypass-vulnerabilities
- regulatory-compliance-drift
- password-security-weaknesses
- poorly-defined-responsibilities
- authorization-role-explosion
layout: solution
lang: de
en_slug: least-privilege
related_solutions:
- slug: authorization
  similarity: 0.75
- slug: authentication
  similarity: 0.75
- slug: security-hardening-process
  similarity: 0.75
- slug: authorization-concept
  similarity: 0.75
- slug: network-segmentation
  similarity: 0.75
- slug: patch-management
  similarity: 0.75
---

## Description

Das Prinzip der geringsten Privilegien (Least Privilege) besagt, dass jedes Nutzerkonto, jedes Dienstkonto und jeder Prozess nur die spezifischen Berechtigungen erhalten sollte, die er zur Erfüllung seiner Funktion braucht, und nicht mehr. Es umzusetzen ist eine Frage systematischer Prüfung bestehenden Zugriffs — Datenbankberechtigungen, Dateisystemberechtigungen, Netzwerkerreichbarkeit, administrative und Sudo-Rechte — und jeden davon auf das erforderliche Minimum zu reduzieren, oft ergänzt durch Just-in-Time-Eskalationsmechanismen, damit breiterer Zugriff nur vorübergehend und auditiert verfügbar ist, wenn er echt gebraucht wird. Legacy-Systeme neigen dazu, über die Zeit in die entgegengesetzte Richtung zu driften: Dienstkonten erhalten administrative Rechte, weil das der schnellste Weg war, ein Feature unter Termindruck zum Laufen zu bringen, einmaliger Debugging-Zugriff wird nie widerrufen, sobald der Vorfall gelöst ist, und Standardkonten, die mit alter Middleware oder Datenbanken ausgeliefert wurden, bleiben aktiv, weil sich niemand mehr an sie erinnert. Das Ergebnis ist, dass eine Anwendungskompromittierung — eine SQL-Injection, ein gestohlenes Credential, ein gekaperter Prozess — weit mehr Reichweite erbt, als die legitime Anwendungslogik je brauchte, was einen eigentlich eingedämmten Vorfall in vollen Zugriff auf unzusammenhängende Daten und Systeme verwandelt. Least Privilege verhindert nicht die anfängliche Kompromittierung, aber es begrenzt ihre Konsequenzen, weshalb es einer der direktesten Wege ist, den Explosionsradius in einer Legacy-Umgebung zu begrenzen, in der die ursprünglichen Zugriffsentscheidungen locker getroffen und nie überarbeitet wurden. Die Kosten der rückwirkenden Anwendung sind, dass Legacy-Anwendungen oft undokumentierte Abhängigkeiten von breiten Berechtigungen haben, sodass das Verschärfen des Zugriffs sorgfältiges Testen erfordert, um nicht Funktionalität zu brechen, die still auf dem Überschuss beruhte.

## How to Apply ◆

> Legacy-Systeme gewähren häufig übermäßige Berechtigungen an Nutzer, Dienstkonten und Prozesse — oft weil es einfacher war, als den minimal erforderlichen Zugriff zu bestimmen. Das Prinzip der geringsten Privilegien beschränkt jede Entität nur auf die für ihre spezifische Funktion notwendigen Berechtigungen.

- Prüfen Sie alle Nutzerkonten und ihre Berechtigungen. Identifizieren Sie Konten mit administrativen oder erhöhten Privilegien und verifizieren Sie, dass jedes Privileg durch die aktuelle Rolle des Kontos gerechtfertigt ist. Legacy-Systeme haben oft Dutzende Konten mit vollem administrativem Zugriff.
- Überprüfen Sie Dienstkontoberechtigungen und reduzieren Sie sie auf das erforderliche Minimum. Legacy-Anwendungsdienstkonten laufen oft als Root/Administrator oder haben vollen Datenbankzugriff, wenn sie nur Zugriff auf bestimmte Tabellen oder Operationen brauchen.
- Implementieren Sie Least Privilege auf Datenbankebene, indem anwendungsspezifische Datenbanknutzer mit granularen Berechtigungen erstellt werden. Ein Reporting-Dienst sollte Nur-Lese-Zugriff haben; ein Transaktionsverarbeiter sollte Lese-Schreib-Zugriff nur auf transaktionsbezogene Tabellen haben.
- Entfernen Sie Standardkonten und -berechtigungen, die mit Legacy-Software, -Datenbanken und -Middleware ausgeliefert werden. Standardkonten mit bekannten Passwörtern sind ein primärer Angriffsvektor.
- Implementieren Sie Just-in-Time-(JIT-)Privilegieneskalation für administrative Aufgaben: Administratoren nutzen Standardkonten für die tägliche Arbeit und eskalieren zu privilegierten Konten nur bei administrativen Operationen, mit automatischem Ablauf.
- Wenden Sie Least Privilege auf Dateisystemberechtigungen an: Anwendungsprozesse sollten nur Zugriff auf die Verzeichnisse haben, die sie brauchen, Konfigurationsdateien sollten nur vom Anwendungsnutzer lesbar sein, und Log-Verzeichnisse sollten nur vom Logging-Prozess beschreibbar sein.
- Überprüfen und beschränken Sie Zugriff auf Netzwerkebene, damit jede Komponente nur mit den spezifischen Endpunkten kommunizieren kann, die sie braucht, statt unbeschränkten Netzwerkzugriff innerhalb des Legacy-System-Segments zu haben.

## Tradeoffs ⇄

> Least Privilege begrenzt den Schaden kompromittierter Konten und verringert die Angriffsfläche, erfordert aber detaillierte Analyse tatsächlicher Zugriffsbedürfnisse und laufende Pflege.

**Vorteile:**

- Begrenzt den Explosionsradius kompromittierter Zugangsdaten — ein Angreifer, der ein beschränktes Konto kompromittiert, kann nur auf die Ressourcen zugreifen, die diesem Konto erlaubt sind.
- Verringert das Risiko versehentlichen Schadens durch administrative Fehler, indem sichergestellt wird, dass Routineoperationen mit begrenzten Berechtigungen laufen.
- Unterstützt Compliance mit Sicherheitsstandards und Vorschriften, die Zugriffskontrolle basierend auf geschäftlichem Bedarf vorschreiben.
- Macht Sicherheitsauditierung effektiver, indem eine klare, dokumentierte Zuordnung zwischen Rollen und Berechtigungen geschaffen wird.

**Kosten und Risiken:**

- Die Bestimmung der minimal erforderlichen Berechtigungen für Legacy-Anwendungen erfordert umfangreiches Testen, da undokumentierte Abhängigkeiten von erhöhten Berechtigungen üblich sind.
- Berechtigungen zu aggressiv zu reduzieren kann Funktionalität brechen, besonders in Legacy-Systemen, in denen die tatsächlichen Berechtigungsanforderungen nicht gut dokumentiert sind.
- Least Privilege erfordert laufende Durchsetzung, während sich das System weiterentwickelt — neue Features könnten neue Berechtigungen brauchen, und alte Berechtigungen könnten widerrufen werden müssen.
- Just-in-Time-Privilegieneskalation fügt administrativen Workflows Reibung hinzu und erfordert unterstützende Infrastruktur.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Least Privilege Risiko in Legacy-Systemen verringert.

Eine Legacy-Webanwendung läuft unter einem Dienstkonto, das vollen administrativen Zugriff auf die SQL-Server-Datenbank hat, einschließlich der Fähigkeit, Tabellen zu erstellen und zu löschen, Schema zu ändern und auf alle Datenbanken auf dem Server zuzugreifen. Als eine SQL-Injection-Schwachstelle ausgenutzt wird, nutzt der Angreifer die Datenbankberechtigungen der Anwendung, um alle Datenbanken aufzuzählen, Daten aus der HR-Datenbank zu extrahieren (die die Anwendung nicht nutzt) und Audit-Tabellen zu löschen, um seine Spuren zu verwischen. Nach der Implementierung von Least Privilege hat der Datenbanknutzer der Anwendung nur SELECT-, INSERT- und UPDATE-Berechtigungen auf die 12 Tabellen, die sie tatsächlich nutzt, ohne Zugriff auf andere Datenbanken, ohne DDL-Berechtigungen und ohne Fähigkeit, Audit-Tabellen zu ändern. Als eine nachfolgende SQL-Injection-Schwachstelle entdeckt wird, ist der Zugriff des Angreifers auf die eigenen Tabellen der Anwendung beschränkt, und die Audit-Spur bleibt intakt, was schnelle Erkennung und Reaktion ermöglicht.

Ein Legacy-Linux-Anwendungsserver hat 15 Nutzerkonten, von denen 8 uneingeschränkten Sudo-Zugriff haben (äquivalent zu Root). Die Untersuchung offenbart, dass 6 dieser Nutzer sich Jahre zuvor für eine einmalige Debugging-Aufgabe selbst zur Sudoers-Datei hinzugefügt und den Zugriff nie entfernt haben. Das Team implementiert ein Least-Privilege-Modell: Sudo-Zugriff wird von allen 6 unnötigen Konten entfernt, die verbleibenden 2 administrativen Konten haben Sudo-Zugriff, beschränkt auf spezifische für ihre Rollen benötigte Befehle, und ein JIT-Privilegieneskalationssystem verlangt von Administratoren, temporären erhöhten Zugriff mit geschäftlicher Begründung und automatischem Ablauf nach 4 Stunden anzufragen. Root-Login wird vollständig deaktiviert, und alle privilegierten Aktionen werden in einem zentralisierten, manipulationssicheren Audit-System protokolliert.
