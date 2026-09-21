---
title: Fuzz-Testing
description: Testen mit zufällig generierten Eingabedaten, um unerwartetes Verhalten
  aufzudecken.
category:
- Security
- Testing
problems:
- buffer-overflow-vulnerabilities
- inadequate-error-handling
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- null-pointer-dereferences
- integer-overflow-underflow
- legacy-code-without-tests
- stack-overflow-errors
layout: solution
lang: de
en_slug: fuzz-testing
related_solutions:
- slug: negative-testing
  similarity: 0.8
- slug: dynamic-code-analysis
  similarity: 0.8
- slug: penetration-tests
  similarity: 0.8
- slug: test-coverage-strategy
  similarity: 0.75
- slug: exploratory-testing
  similarity: 0.7
- slug: input-validation
  similarity: 0.7
---

## Description

Fuzz-Testing versorgt ein Programm mit großen Mengen zufällig generierter oder systematisch mutierter Eingabe und überwacht auf Abstürze, Hänger, Speicherbeschädigung oder anderes anomales Verhalten, ohne vorherige Annahme darüber, wie die korrekte Ausgabe aussehen sollte. Dies macht es grundlegend anders als konventionelles Testen, das prüft, ob bekannte Eingaben erwartete Ausgaben produzieren: Ein Fuzzer sucht stattdessen nach den Eingaben, die das Programm zum Brechen bringen, und entdeckt Fehlermodi, die kein Testautor antizipiert hat. Legacy-Systeme sind besonders fruchtbarer Boden für diese Technik, weil ihr Eingabeverarbeitungscode — benutzerdefinierte Parser, Protokoll-Handler, Dateiformat-Leser — häufig Jahrzehnte zuvor unter Bedrohungsmodellen geschrieben wurde, die nicht mehr gelten, und seither oft nie mit fehlerhafter, überdimensionierter oder feindlicher Eingabe erprobt wurde. Mutationsbasierte Fuzzer, die den laufenden Code instrumentieren, können tiefe und ungewöhnliche Ausführungspfade erkunden, wenn Quellcode verfügbar ist, während Black-Box-Fuzzer weiterhin externe Schnittstellen von Legacy-Komponenten sondieren können, deren Interna undurchsichtig oder undokumentiert sind. Weil Fuzzing keine Spezifikation korrekten Verhaltens erfordert, umgeht es das zentrale Hindernis beim Testen von Legacy-Code — die Abwesenheit von Dokumentation — und zielt stattdessen direkt auf die Klassen von Schwachstellen, wie Pufferüberläufe und Integer-Überläufe, die am wahrscheinlichsten unbemerkt in alten, selten angefassten Eingabepfaden überlebt haben.

## How to Apply ◆

> Legacy-Systeme enthalten oft Eingabeverarbeitungscode, der nie mit unerwarteten, fehlerhaften oder feindlichen Eingaben getestet wurde. Fuzz-Testing generiert systematisch zufällige und semi-zufällige Eingaben, um Abstürze, Hänger, Speicherfehler und anderes unerwartetes Verhalten zu entdecken, die auf Sicherheitslücken hindeuten.

- Identifizieren Sie Fuzzing-Ziele im Legacy-System: Eingabeparser (Dateiformate, Netzwerkprotokolle, API-Payloads), Deserialisierungsroutinen, Kommandozeilenargumentverarbeitung und jeglichen Code, der nicht vertrauenswürdige externe Eingabe verarbeitet.
- Beginnen Sie mit mutationsbasiertem Fuzzing für Legacy-Systeme, bei denen Quellcode verfügbar ist: Nutzen Sie Werkzeuge wie AFL++, libFuzzer oder Jazzer, die die Anwendung instrumentieren und gültige Eingaben mutieren, um Codepfade zu erkunden, die normales Testen nicht erreicht.
- Nutzen Sie für Legacy-Komponenten ohne Quellcode Black-Box-Fuzzing-Werkzeuge, die randomisierte Eingaben an die externen Schnittstellen der Anwendung (HTTP-Endpunkte, Netzwerk-Sockets, Dateieingaben) senden und auf Abstürze, Fehlerantworten und anomales Verhalten überwachen.
- Implementieren Sie korpusbasiertes Fuzzing, indem Sie reale Eingaben (Produktions-Anfrageprotokolle, Beispieldateien, Protokoll-Aufzeichnungen) als initialen Seed-Korpus sammeln. Mutationsbasierte Fuzzer sind effektiver, wenn sie mit gültigen Eingaben beginnen, die sinnvolle Codepfade ausüben.
- Konfigurieren Sie Absturz-Triage und -Deduplizierung, um das Volumen der Befunde zu verwalten. Fuzzer entdecken oft Hunderte von Abstürzen, die sich auf eine Handvoll eindeutiger Grundursachen reduzieren — automatisierte Deduplizierung verhindert verschwendeten Untersuchungsaufwand.
- Führen Sie Fuzzing-Kampagnen kontinuierlich statt als einmalige Tests durch. Fuzzer entdecken über die Zeit tiefere Bugs, während sie mehr Codepfade erkunden, und neue Codeänderungen könnten neue Schwachstellen einführen.
- Integrieren Sie Fuzz-Testing in CI/CD für kritische Eingabeverarbeitungskomponenten und führen Sie kurze Fuzzing-Sitzungen (10-30 Minuten) bei jedem Build durch, um Regressionen früh zu fangen.

## Tradeoffs ⇄

> Fuzz-Testing entdeckt Eingabeverarbeitungs-Schwachstellen, die andere Testmethoden übersehen, erfordert aber Rechenressourcen, produziert Ergebnisse, die Expertenanalyse benötigen, und ist möglicherweise nicht auf alle Legacy-System-Komponenten anwendbar.

**Vorteile:**

- Entdeckt Randfall-Schwachstellen (Pufferüberläufe, Integer-Überläufe, Nullzeiger-Dereferenzierungen), die Entwickler und konventionelles Testen nicht antizipieren.
- Erfordert kein Vorwissen über das erwartete Verhalten der Anwendung — der Fuzzer entdeckt, was die Anwendung zum Scheitern bringt, statt zu testen, was funktionieren sollte.
- Kann auf Legacy-Code angewendet werden, ohne umfangreiche Test-Harness-Konstruktion, besonders für Black-Box-Fuzzing externer Schnittstellen.
- Liefert reproduzierbare Absturz-Eingaben, die sowohl als Beleg der Schwachstelle als auch als Regressionstestfälle nach der Behebung dienen.

**Kosten und Risiken:**

- Fuzzing erfordert erhebliche Rechenressourcen, wenn es über längere Zeiträume läuft, besonders für abdeckungsgesteuerte Fuzzer, die die Anwendung instrumentieren.
- Absturz-Triage erfordert Sicherheitsexpertise, um zu bestimmen, welche Abstürze ausnutzbare Schwachstellen und welche harmlose Fehlschläge sind.
- Black-Box-Fuzzing von Legacy-Anwendungen kann Instabilität, Datenbeschädigung oder Ressourcenerschöpfung im Zielsystem verursachen, was isolierte Testumgebungen erfordert.
- Manche Schwachstellenklassen (Logikfehler, Autorisierungsumgehungen, Geschäftslogikfehler) sind durch Eingabe-Fuzzing nicht entdeckbar.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Fuzz-Testing Schwachstellen in Legacy-Systemen aufdeckt.

Ein Legacy-Dateiverarbeitungssystem akzeptiert XML-Dateien, die von Geschäftspartnern für die Auftragsverarbeitung hochgeladen werden. Der in C geschriebene XML-Parser, seit 15 Jahren unverändert, wurde nie mit fehlerhafter Eingabe getestet. Das Team führt AFL++ gegen den Parser aus, unter Nutzung eines Korpus von 200 echten XML-Auftragsdateien als Seeds. Nach 48 Stunden Fuzzing entdeckt das Werkzeug 7 eindeutige Abstürze: 3 Pufferüberläufe, ausgelöst durch überdimensionierte Elementnamen, 2 Nullzeiger-Dereferenzierungen durch fehlerhafte Namespace-Deklarationen, 1 Integer-Überlauf im Elementtiefenzähler und 1 Stack-Overflow durch tief verschachtelte Elemente. Zwei der Pufferüberläufe werden als ausnutzbar für Remote-Code-Ausführung bestätigt. Das Team behebt alle sieben Probleme und fügt die abstürzenden Eingaben als permanente Regressionstests hinzu. Sie implementieren zudem eine speichersichere XML-Parsing-Bibliothek, um den benutzerdefinierten Parser zu ersetzen.

Ein Legacy-Netzwerkdienst verarbeitet Binärprotokollnachrichten von Industriecontrollern. Die Protokollspezifikation ist teilweise dokumentiert, und der Parsing-Code enthält viele unvalidierte Annahmen über die Nachrichtenstruktur. Das Team konstruiert eine einfache Fuzzing-Harness, die randomisierte Binärdaten an den Netzwerk-Port des Dienstes sendet und auf Abstürze und Hänger überwacht. Innerhalb von 6 Stunden entdeckt der Fuzzer, dass eine Nachricht mit einem Längenfeld von null den Parser in eine Endlosschleife bringt, die 100 % CPU verbraucht. Eine weitere fehlerhafte Nachricht verursacht einen Heap-Pufferüberlauf, wenn die angegebene Payload-Länge die tatsächliche Nachrichtengröße überschreitet. Beide Probleme könnten von jedem Gerät im Industrienetzwerk ausgenutzt werden, um den Dienst zum Absturz zu bringen oder zu kompromittieren. Das Team fügt explizite Längenvalidierung und Eingabegrenzenprüfung hinzu und behebt Schwachstellen, die seit der Erstellung des Protokoll-Handlers vor 12 Jahren existierten.
