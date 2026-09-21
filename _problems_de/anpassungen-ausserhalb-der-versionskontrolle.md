---
title: Anpassungen außerhalb der Versionskontrolle
description: Konfiguration und individuelle Logik liegen in der Datenbank eines
  kommerziell erworbenen Softwaresystems, sodass sie nicht diffbar, überprüfbar,
  reproduzierbar oder auf ihren Urheber zurückverfolgbar sind.
category:
- Operations
- Process
- Code
related_problems:
- slug: excessive-customization
  similarity: 0.7
- slug: low-code-customization-sprawl
  similarity: 0.65
- slug: inadequate-configuration-management
  similarity: 0.65
- slug: custom-report-sprawl
  similarity: 0.65
- slug: core-modification-of-standard-software
  similarity: 0.6
- slug: reimplemented-standard-functionality
  similarity: 0.6
solutions:
- customization-under-version-control
- version-control
- ci-cd-pipeline
- infrastructure-as-code
- code-reviews
- environment-parity
- configuration-checks
- immutable-infrastructure
- clear-ownership-model
- audit-trail-management
- role-model-rationalization
layout: problem
lang: de
en_slug: customization-outside-version-control
---

## Description

In vielen kommerziell erworbenen Softwaresystemen werden die Anpassungen – Konfiguration, benutzerdefinierte Felder, Workflow-Definitionen, Skripte, Report-Layouts, Rollenzuweisungen – in der eigenen Datenbank des Produkts gespeichert statt als Dateien. Es gibt kein Repository, keinen Commit, kein Diff und oft keine Aufzeichnung, wer was geändert hat oder warum. Die Konsequenz ist, dass jede auf Versionskontrolle aufbauende Engineering-Praxis schlicht nicht anwendbar ist: Änderungen können nicht überprüft werden, bevor sie wirksam werden, eine Umgebung kann nicht aus einem bekannten Zustand reproduziert werden, eine Änderung kann nicht zurückgenommen werden, außer indem man sich erinnert, wie sie war, und der gesamte Anpassungsbestand kann nicht aufgelistet werden. Teams, die in ihren eigenen Codebasen strenge Disziplin pflegen, betreiben diese Systeme häufig ohne all das, ohne die Inkonsistenz zu bemerken, weil die Werkzeuge diese Option nie geboten haben.

## Indicators ⟡

- Niemand kann beantworten, was sich im System im letzten Monat geändert hat, ohne Leute zu fragen
- Test- und Produktionskonfigurationen unterscheiden sich auf Weisen, die entdeckt statt gewusst werden
- Eine Änderung wird direkt in der Produktion vorgenommen, weil dort die Konfiguration liegt und es keinen anderen Weg gibt, sie anzuwenden
- Das Zurücknehmen einer Änderung bedeutet, dass jemand den vorherigen Wert aus dem Gedächtnis oder einem Screenshot rekonstruiert
- Es gibt keinen Review-Schritt, bevor eine Konfigurationsänderung wirksam wird, und keine Aufzeichnung, dass eine stattgefunden hat
- Das Nachbauen einer funktionierenden Umgebung von Grund auf gilt als unmöglich oder dauert Wochen manuellen Vergleichs
- Die Anzahl benutzerdefinierter Objekte, Felder oder Workflows ist unbekannt und kann nur durch Export und Zählen festgestellt werden

## Symptoms ▲

- [Konfigurations-Drift](konfigurations-drift.md)
<br/>  Ohne eine einzige maßgebliche Quelle entwickeln sich Umgebungen kontinuierlich auseinander, und die Abweichung wird erst gefunden, wenn sich etwas anders verhält.
- [Manuelle Deployment-Prozesse](manuelle-deployment-prozesse.md)
<br/>  Eine Änderung zwischen Umgebungen zu verschieben bedeutet, sie von Hand zu wiederholen, was langsam, fehleranfällig und nicht verifizierbar ist.
- [Unsichtbarkeit technischer Schulden](unsichtbarkeit-technischer-schulden.md)
<br/>  Anpassung, die nicht aufgelistet werden kann, kann nicht bewertet werden, sodass ihr angehäuftes Gewicht für alle unsichtbar bleibt, einschließlich der Personen, die es tragen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Änderungen werden ohne Review und ohne Rückweg wirksam, sodass eine fehlerhafte Anpassung Nutzer erreicht und bestehen bleibt, bis jemand den vorherigen Zustand rekonstruiert.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Ohne eine Aufzeichnung der Urheberschaft hat keine Änderung einen Verantwortlichen, und Fragen dazu, warum etwas auf eine bestimmte Weise konfiguriert ist, haben keinen Adressaten.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Die Konfiguration ist ihre eigene Dokumentation, und sie ist nicht lesbar, sodass das Verständnis des Systems das Durchklicken von Bildschirmen erfordert statt Lesen.
- [Wissenssilos](wissenssilos.md)
<br/>  Was das System tut, wissen nur diejenigen, die die Änderungen vorgenommen haben, weil die Änderungen kein Artefakt hinterlassen haben, das jemand anderes lesen kann.
- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Die Diagnose kann nicht bei dem beginnen, was sich kürzlich geändert hat, da diese Frage keine Antwort hat, sodass Untersuchungen jedes Mal von vorne beginnen.

## Causes ▼

- [Vendor Lock-in](vendor-lock-in.md)
<br/>  Das Produkt speichert seine Konfiguration von Natur aus intern, und sie in eine überprüfbare Form zu exportieren erfordert Aufwand, den der Hersteller nicht unterstützt.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Mitarbeitende, die das System administrieren, kommen oft aus einem Betriebs- statt einem Entwicklungshintergrund, und Versionskontrolle ist nicht Teil der Praxis, die sie gelernt haben.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Eine Export- und Deployment-Pipeline aufzubauen liefert nichts Sichtbares, während die direkte Änderung es heute liefert.

## Detection Methods ○

- Um ein Diff der Konfiguration zwischen zwei Umgebungen bitten; die Schwierigkeit, eines zu erstellen, misst das Problem direkt
- Versuch zu beantworten, was sich in den letzten dreißig Tagen geändert hat, und wie lange die Antwort dauert
- Prüfung, ob jede Änderung an der Produktkonfiguration ein Review durchläuft, bevor sie wirksam wird
- Versuch, eine Testumgebung aus einer definierten Quelle nachzubauen, und Aufzeichnung, was von Hand getan werden muss
- Prüfung, ob das Produkt ein Exportformat anbietet und ob jemand es nutzt
- Zählung, wie viele Personen die Produktionskonfiguration direkt ändern können und ob ihre Änderungen protokolliert werden

## Examples

Eine IT-Service-Management-Plattform war seit sechs Jahren im Einsatz, konfiguriert von vier Administratoren über drei Teams. Workflows, Formularlogik, Genehmigungsregeln und mehrere hundert skriptgesteuerte Verhalten lagen vollständig in der Datenbank der Plattform. Als eine Änderung an einer Genehmigungsweiterleitung begann, Anfragen an eine nicht mehr existierende Abteilung zu senden, konnte die Untersuchung nicht feststellen, wann die Weiterleitung eingerichtet worden war, von wem oder wie sie vorher war. Drei Administratoren erinnerten sich jeweils an einen anderen ursprünglichen Wert. Die Lösung dauerte elf Tage, wovon der Großteil damit verbracht wurde, die Absicht zu rekonstruieren, statt die Änderung vorzunehmen. Der Anwendungscode der Organisation dagegen wurde überprüft, getestet und über eine Pipeline deployt – dieselben Ingenieure hatten schlicht kein Äquivalent für die Plattform, die ihren Vorfallprozess betrieb.

Die Kosten der Reproduzierbarkeit kamen bei einer Disaster-Recovery-Übung zum Vorschein. Der Plan ging davon aus, dass eine Ersatzinstanz aus Dokumentation gebaut und konfiguriert werden könnte. Die Übung stellte fest, dass die Dokumentation etwa ein Drittel der Live-Konfiguration beschrieb, dass der Rest nur im laufenden System existierte, und dass die tatsächliche Wiederherstellungsposition der Organisation vollständig davon abhing, dass das Datenbank-Backup wiederherstellbar war. Niemand hatte dies als Anpassungsproblem betrachtet; es war vier Jahre lang als Infrastrukturangelegenheit abgelegt worden.
