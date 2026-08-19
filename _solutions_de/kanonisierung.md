---
title: Kanonisierung
description: Transformation von Eingabedaten in eine kanonische Darstellung.
category:
- Security
- Code
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- inconsistent-behavior
- buffer-overflow-vulnerabilities
- log-injection-vulnerabilities
- inadequate-error-handling
layout: solution
lang: de
en_slug: canonicalization
related_solutions:
- slug: input-validation
  similarity: 0.8
- slug: output-encoding
  similarity: 0.75
- slug: authentication
  similarity: 0.7
- slug: encryption
  similarity: 0.7
- slug: data-flow-control
  similarity: 0.7
- slug: defense-lines
  similarity: 0.65
---

## Description

Kanonisierung ist der Prozess, Eingaben in eine einzige, wohldefinierte Standardrepräsentation zu transformieren, bevor diese Eingabe validiert, verglichen oder verarbeitet wird, sodass Sicherheitsprüfungen und Geschäftslogik auf einer vorhersehbaren Form operieren statt auf irgendeiner ihrer vielen äquivalenten Kodierungen. Daten kommen häufig mit derselben zugrunde liegenden Bedeutung auf unterschiedliche Weise ausgedrückt an — URL-kodierte oder doppelt kodierte Zeichen, unterschiedliche Unicode-Normalisierungsformen, Pfadausdrücke mit redundanten Trennzeichen oder Traversal-Sequenzen — und Kanonisierung reduziert diese Varianten auf eine Form, bevor irgendetwas Nachgelagertes sie prüft. Dies ist für Legacy-Systeme akut wichtig, weil ihre Eingabefilter und Validierungsroutinen oft für eine einzige Kodierungsannahme geschrieben wurden und nie aktualisiert wurden, als neue Kodierungspfade auf Netzwerk- oder Anwendungsebene hinzugefügt wurden, was eine Lücke zwischen dem, was ein Filter prüft, und dem, was die Anwendung tatsächlich verarbeitet, hinterlässt. Angreifer nutzen genau diese Lücke aus, indem sie einen Payload in einer Kodierung einreichen, die der Filter nicht als gefährlich erkennt, in der Hoffnung, dass eine nachgelagerte Komponente ihn in die schädliche Form dekodiert, nachdem die Prüfung bereits bestanden wurde. Indem zuerst normalisiert und die kanonische Form statt der Rohdeingabe validiert wird, schließt Kanonisierung diese Umgehungsklasse systematisch, statt zu verlangen, dass jeder Filter jeden möglichen Kodierungstrick antizipiert. Es vereinfacht außerdem die Validierungslogik selbst, da Regeln nur eine Repräsentation statt einer offenen Menge äquivalenter berücksichtigen müssen, was in Legacy-Codebasen wertvoll ist, wo Validierung oft inkonsistent über viele Einstiegspunkte dupliziert ist. Weil Kanonisierung Daten verändern kann, wenn sie falsch angewendet wird, muss sie sorgfältig in Systemen implementiert werden, deren interne Logik bereits auf spezifischen nicht-kanonischen Formen beruht.

## How to Apply ◆

> Legacy-Systeme verarbeiten Eingaben oft in mehreren Kodierungen und Formaten, ohne sie zuerst zu normalisieren, was Angreifern Gelegenheiten schafft, Sicherheitsfilter mit kodierten oder verschleierten Payloads zu umgehen. Kanonisierung transformiert alle Eingaben in eine einzige Standardform, bevor Validierung und Verarbeitung stattfinden.

- Identifizieren Sie alle Eingabeeinstiegspunkte im Legacy-System, wo Daten in variablen Formaten ankommen: URLs, Dateipfade, Zeichenkodierungen, Unicode-Repräsentationen, HTML-Entitäten und URL-kodierte Werte.
- Wenden Sie Kanonisierung als ersten Schritt in der Eingabeverarbeitung an, vor jeglichen Sicherheitsprüfungen oder Validierungen. Validieren Sie gegen die kanonische Form, nicht die Rohdeingabe — Angreifer nutzen die Lücke zwischen dem, was der Sicherheitsfilter sieht, und dem, was die Anwendung verarbeitet.
- Normalisieren Sie Unicode-Eingaben auf eine konsistente Form (NFC oder NFKC), um Angriffe mit visuell identischen, aber technisch unterschiedlichen Zeichenfolgen zu verhindern. Legacy-Systeme handhaben Unicode-Normalisierung oft nicht, was Homograph-Angriffe und Filterumgehungen erlaubt.
- Lösen Sie alle Pfadkomponenten (Punkt-Punkt-Sequenzen, symbolische Links, redundante Trennzeichen) zu absoluten kanonischen Pfaden auf, bevor Sie Zugriffsberechtigungen prüfen. Dies verhindert Path-Traversal-Angriffe, die kodierte Verzeichnistraversal-Sequenzen nutzen.
- Dekodieren Sie alle Kodierungsschichten (URL-Kodierung, HTML-Entitäten, Base64, doppelte Kodierung) vollständig, bevor Sie Validierungsregeln anwenden. Viele Legacy-Sicherheitsfilter prüfen nur die erste Kodierungsschicht, während die Anwendung mehrere Schichten dekodiert.
- Standardisieren Sie Datenformate (Daten, Zahlen, Kennungen) in eine einzige kanonische Repräsentation an der Systemgrenze, um Inkonsistenzen zu verhindern, die zu Logikfehlern und Sicherheitsumgehungen führen.
- Implementieren Sie Kanonisierung in einer gemeinsam genutzten Utility-Bibliothek, sodass alle Eingabeverarbeitungspfade dieselbe Normalisierungslogik nutzen, was Inkonsistenzen zwischen verschiedenen Teilen der Codebasis verhindert.

## Tradeoffs ⇄

> Kanonisierung eliminiert kodierungsbasierte Sicherheitsumgehungen, indem sichergestellt wird, dass alle Eingaben in einer bekannten, standardisierten Form vorliegen, bevor Validierung stattfindet, erfordert aber umfassende Identifikation aller Kodierungsschemata und sorgfältige Implementierung.

**Vorteile:**

- Verhindert Sicherheitsfilterumgehungen mithilfe alternativer Kodierungen, doppelter Kodierung und Unicode-Tricks, die Unterschiede zwischen der Sicht des Filters und der Sicht der Anwendung auf die Eingabe ausnutzen.
- Verringert die Komplexität von Validierungsregeln, indem sichergestellt wird, dass sie nur eine kanonische Form statt mehrerer äquivalenter Repräsentationen handhaben müssen.
- Verbessert die Datenkonsistenz, indem Eingaben an der Systemgrenze auf eine Standardform normalisiert werden.
- Macht Sicherheitstests effektiver, weil die kanonische Form vorhersehbar ist und systematisch validiert werden kann.

**Kosten und Risiken:**

- Falsche Kanonisierung kann die semantische Bedeutung der Eingabe verändern, was Datenkorruption oder funktionale Fehler verursacht.
- Legacy-Systeme könnten sich intern auf spezifische nicht-kanonische Repräsentationen verlassen, was Kanonisierung an der Grenze mit bestehender Verarbeitungslogik inkompatibel macht.
- Übermäßig aggressive Kanonisierung (Entfernen oder Ersetzen von Zeichen) kann legitime internationale Eingaben ablehnen oder beschädigen.
- Der Performance-Overhead durch Kanonisierung ist typischerweise gering, kann aber bei hochvolumiger Eingabeverarbeitung in Legacy-Systemen spürbar sein.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Kanonisierung Sicherheitsumgehungen in Legacy-Systemen verhindert.

Eine Legacy-Webanwendung hat einen Eingabefilter, der SQL-Injection blockiert, indem er nach der Zeichenkette „SELECT" in Formulareinreichungen sucht. Ein Angreifer umgeht diesen Filter, indem er die Abfrage mit URL-kodierten Zeichen einreicht: „%53ELECT". Der Webserver der Anwendung dekodiert die URL-Kodierung, bevor er sie an die Anwendung weitergibt, sodass die Anwendung „SELECT" verarbeitet, während der Filter „%53ELECT" sah und es durchließ. Das Team implementiert Kanonisierung, indem eine Middleware-Schicht hinzugefügt wird, die alle URL-Kodierung, HTML-Entitäten und Unicode-Escapes vollständig dekodiert, bevor die Eingabe den Sicherheitsfilter erreicht. Nach der Kanonisierung sieht der Filter „SELECT" unabhängig davon, wie der Angreifer es kodiert, und der Injection-Versuch wird blockiert. Das Team ersetzt außerdem den einfachen String-Matching-Filter durch parametrisierte Abfragen, wobei Kanonisierung als zusätzliche Verteidigungsschicht genutzt wird.

Eine Legacy-Dateifreigabeanwendung erlaubt Nutzern, Dateien durch Angabe eines Dateipfadparameters herunterzuladen. Die Anwendung prüft, dass der Pfad kein „.." enthält, um Directory Traversal zu verhindern. Ein Angreifer nutzt die URL-kodierte Form „%2e%2e%2f", um Verzeichnisse zu durchqueren und auf die Passwortdatei des Systems zuzugreifen. Nach der Implementierung von Pfadkanonisierung, die alle kodierten Sequenzen auflöst und Pfade in ihre absolute kanonische Form umwandelt, bevor die Sicherheitsprüfung stattfindet, identifiziert die Anwendung den Traversal-Versuch korrekt und lehnt ihn ab. Der kanonisierte Pfad „/var/data/../../etc/passwd" wird zu „/etc/passwd", was klar die Prüfung nicht besteht, dass alle zugegriffenen Dateien unter „/var/data/" liegen müssen.
