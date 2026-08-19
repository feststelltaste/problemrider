---
title: Negativtests
description: Gezieltes Testen ungültiger Eingaben und Grenzfälle zur Prüfung der
  Fehlerbehandlung.
category:
- Security
- Testing
problems:
- inadequate-error-handling
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- insufficient-testing
- legacy-code-without-tests
- error-message-information-disclosure
- null-pointer-dereferences
- integer-overflow-underflow
- stack-overflow-errors
- race-conditions
layout: solution
lang: de
en_slug: negative-testing
related_solutions:
- slug: fuzz-testing
  similarity: 0.8
- slug: test-coverage-strategy
  similarity: 0.75
- slug: input-validation
  similarity: 0.75
- slug: functional-tests
  similarity: 0.75
- slug: penetration-tests
  similarity: 0.75
- slug: automated-tests
  similarity: 0.75
---

## Description

Negativtests sind die Praxis, ein System bewusst mit ungültiger, fehlerhafter, grenzwertiger oder bösartiger Eingabe zu füttern und zu bestätigen, dass es sicher fehlschlägt, statt nur den „Happy Path" auszuüben, den gültige Eingaben durchlaufen. Wo funktionales Testen bestätigt, dass ein Feature das Richtige tut, wenn es angemessene Eingabe erhält, prüft Negativtesten, was das System tut, wenn es Eingabe erhält, für die es nie entworfen wurde — leere Werte, überdimensionierte Payloads, falsche Typen, außerhalb des Bereichs liegende Zahlen und bekannte Angriffsmuster wie Injection-Strings. In Legacy-Systemen zählt diese Unterscheidung, weil sich Jahrzehnte inkrementeller Feature-Arbeit dazu neigen, Testaufwand darauf zu konzentrieren zu zeigen, dass neue Funktionalität funktioniert, während sich Fehlerpfade still ansammeln und selten ausgeübt werden, bis ein Angreifer oder ein ungewöhnliches Produktionsereignis sie schließlich erreicht. Weil Legacy-Fehlerbehandlung häufig ad hoc implementiert wird — String-Verkettung in Abfragen, rohe Exception-Meldungen, die Nutzern angezeigt werden, fehlende Grenzprüfungen — enthalten diese Pfade unverhältnismäßig häufig genau die Injection-, Offenlegungs- und Absturz-Schwachstellen, die Negativtesten ans Licht bringen soll. Die Technik ist vergleichsweise günstig nachzurüsten: Sie erfordert kein Neuschreiben von Geschäftslogik, nur das Hinzufügen von Testfällen, die feindliche Eingabe an bereits bestehende Schnittstellen senden. Einmal entdeckt, werden Befunde typischerweise lokal behoben (Parametrisierung, Validierung, generische Fehlermeldungen) und dann als automatisierte Regressionstests festgeschrieben, damit derselbe Defekt nicht still zurückkehren kann, während sich das Legacy-System weiterentwickelt.

## How to Apply ◆

> Legacy-System-Tests verifizieren typischerweise, dass das System mit gültigen Eingaben korrekt funktioniert, testen aber nie, was mit ungültigen, unerwarteten oder bösartigen Eingaben passiert. Negativtesten liefert bewusst schlechte Eingaben, um zu verifizieren, dass die Fehlerbehandlung korrekt und sicher ist.

- Definieren und testen Sie für jedes Eingabefeld und jeden API-Parameter Grenzwerte: Maximallänge + 1, Minimalwert - 1, leere Zeichenfolgen, Null-Werte, negative Zahlen, wo positive erwartet werden, und Werte außerhalb aufgezählter Mengen.
- Testen Sie mit Eingaben, die bekannte Angriffsmuster sind: SQL-Injection-Payloads, XSS-Skripte, Path-Traversal-Sequenzen, Command-Injection-Strings und Format-String-Spezifizierer. Das System sollte diese sauber handhaben, ohne den injizierten Inhalt auszuführen.
- Verifizieren Sie Fehlerantworten: Das System sollte angemessene Fehlermeldungen zurückgeben, die dem Nutzer helfen, seine Eingabe zu korrigieren, ohne Implementierungsdetails (Stack Traces, Datenbankfehler, Dateipfade, Versionsnummern) offenzulegen, die einem Angreifer helfen würden.
- Testen Sie Authentifizierung und Autorisierung negativ: Versuchen Sie Zugriff ohne Zugangsdaten, mit abgelaufenen Zugangsdaten, mit den Zugangsdaten eines anderen Nutzers und mit modifizierten Tokens. Verifizieren Sie, dass jedes Szenario ordentlich abgelehnt wird.
- Testen Sie nebenläufige und außer der Reihe erfolgende Operationen: Senden Sie Anfragen in unerwarteten Sequenzen, senden Sie doppelte Anfragen und testen Sie Race Conditions, indem Sie konkurrierende Modifikationen gleichzeitig einreichen.
- Testen Sie Ressourcenbegrenzungen: Laden Sie Dateien hoch, die Größenbegrenzungen überschreiten, senden Sie Anfragen mit Raten über Rate-Limits, erstellen Sie Objekte, die Mengenbegrenzungen überschreiten. Verifizieren Sie, dass Begrenzungen durchgesetzt und überschrittene Anfragen sauber gehandhabt werden.
- Implementieren Sie Negativtestfälle als automatisierte Tests, die in der CI/CD-Pipeline laufen, um sicherzustellen, dass die Fehlerbehandlung korrekt bleibt, während sich das Legacy-System weiterentwickelt.

## Tradeoffs ⇄

> Negativtesten verifiziert, dass Fehlerbehandlung korrekt und sicher ist, was verhindert, dass Angreifer Fehlerbedingungen ausnutzen, erfordert aber kreatives Testdesign und umfassende Abdeckung.

**Vorteile:**

- Entdeckt Fehlerbehandlungsfehler, die das System Injection-Angriffen, Informationsoffenlegung und Denial of Service aussetzen.
- Verifiziert, dass das System sicher fehlschlägt statt offen zu versagen, wenn es unerwarteter Eingabe ausgesetzt wird.
- Fängt sicherheitsrelevante Fehlerpfade ab, die funktionales Testen nie ausübt, und schließt Lücken in der Testabdeckung.
- Bietet Sicherheit, dass Änderungen an Eingabevalidierung und Fehlerbehandlung keine Regressionen einführen.

**Kosten und Risiken:**

- Das Entwerfen umfassender Negativtestfälle erfordert Kreativität und Sicherheitswissen, um die Bandbreite ungültiger Eingaben zu antizipieren, die ein Angreifer nutzen könnte.
- Negativtests können brüchig sein, wenn sie von spezifischem Fehlermeldungstext oder Fehlercodewerten abhängen, die sich über Versionen hinweg ändern.
- Das Ausführen aggressiver Negativtests gegen Legacy-Systeme kann Abstürze oder Datenbeschädigung in Testumgebungen verursachen, was sorgfältige Isolation erfordert.
- Der Raum möglicher ungültiger Eingaben ist unendlich; Negativtesten verringert Risiko, kann aber nicht alle Fehlerbehandlungs-Schwachstellen beseitigen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Negativtesten Sicherheitsschwächen in Legacy-Systemen aufdeckt.

Das Registrierungsformular einer Legacy-Webanwendung validiert, dass E-Mail-Adressen ein „@"-Zeichen enthalten, führt aber keine weitere Validierung durch. Negativtesten offenbart, dass das Einreichen einer E-Mail-Adresse `admin'--@example.com` einen SQL-Fehler verursacht, weil der Wert ohne Parametrisierung in eine Abfrage verkettet wird, und die Fehlermeldung zeigt die vollständige SQL-Abfrage einschließlich Tabellen- und Spaltennamen an. Weitere Negativtests entdecken, dass das Einreichen einer 10.000-Zeichen-E-Mail-Adresse einen Pufferüberlauf in der E-Mail-Validierungsfunktion verursacht, und das Einreichen einer E-Mail mit `<script>alert(1)</script>` dazu führt, dass das Skript ausgeführt wird, wenn der Administrator die Nutzerliste betrachtet. Jeder Befund führt zu einer spezifischen Korrektur: parametrisierte Abfragen, Eingabelängenvalidierung und Ausgabekodierung. Die Negativtestfälle werden automatisiert und laufen bei jedem Build, um Regression zu verhindern.

Eine Legacy-REST-API akzeptiert JSON-Payloads für die Auftragserstellung. Funktionale Tests verifizieren, dass wohlgeformte Aufträge korrekt verarbeitet werden, aber Negativtesten offenbart mehrere Probleme: Das Senden eines JSON-Payloads mit negativer Menge führt dazu, dass das System eine Gutschrift statt einer Belastung erstellt, das Senden eines Preisfeldes mit 15 Dezimalstellen verursacht einen Gleitkommagenauigkeitsfehler, der ausgenutzt werden kann, um im großen Maßstab um Bruchteile eines Cents zu unterzahlen, und das Senden einer nicht existierenden Produkt-ID gibt einen 500-Fehler mit einem Stack Trace zurück, der Datenbankverbindungszeichenfolgen-Details enthält. Das Team behebt jedes Problem (Mengenvalidierung, Dezimalgenauigkeitsbehandlung, generische Fehlerantworten) und fügt der automatisierten Testsuite 35 Negativtestfälle hinzu, die ungültige Typen, außerhalb des Bereichs liegende Werte, fehlende Pflichtfelder und bekannte Angriffsmuster für jeden API-Parameter abdecken.
