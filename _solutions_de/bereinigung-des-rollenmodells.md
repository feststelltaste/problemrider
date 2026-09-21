---
title: Bereinigung des Rollenmodells
description: Wiederaufbau eines explodierten Berechtigungsmodells aus
  Evidenz darüber, was Menschen tatsächlich tun, und Etablierung eines
  Prozesses, durch den Zugriff sowohl entzogen als auch gewährt wird.
category:
- Security
- Operations
- Process
problems:
- authorization-role-explosion
- authorization-flaws
- regulatory-compliance-drift
- increased-manual-work
- lack-of-ownership-and-accountability
- invisible-nature-of-technical-debt
- excessive-customization
- slow-incident-resolution
- customization-outside-version-control
- user-frustration
layout: solution
lang: de
en_slug: role-model-rationalization
related_solutions:
- slug: authorization-concept
  similarity: 0.7
- slug: least-privilege
  similarity: 0.65
- slug: authorization
  similarity: 0.65
- slug: clear-ownership-model
  similarity: 0.65
- slug: clear-roles-and-ownership
  similarity: 0.6
- slug: feature-usage-measurement
  similarity: 0.6
---

## Description

Bereinigung des Rollenmodells rekonstruiert ein ausgeuferten Berechtigungsmodell aus Evidenz — was Nutzer tatsächlich tun, statt was ihnen gewährt wurde — und paart die Rekonstruktion mit einem Prozess, der Zugriff sowohl entfernt als auch hinzufügt. Beide Hälften sind notwendig. Bereinigung ohne Prozessänderung produziert ein sauberes Modell, das innerhalb weniger Jahre auf seine vorherige Größe zurückwächst, da der Mechanismus, der die Explosion verursachte, unangetastet bleibt. Prozessänderung ohne Bereinigung lässt das bestehende Sediment für immer an Ort und Stelle. Die Rekonstruktion ist möglich, weil die meisten Systeme aufzeichnen, was tatsächlich genutzt wurde, und die Lücke zwischen gewährten und genutzten Berechtigungen typischerweise enorm ist. Diese Lücke ist es, was die Arbeit handhabbar macht: Das Zielmodell wird aus beobachtetem Verhalten abgeleitet statt aus einem Organigramm entworfen, das ohnehin nie der Realität entsprach.

## How to Apply ◆

> Der Grund, warum niemand eine Berechtigung entfernt, ist, dass niemand feststellen kann, wer von ihr abhängt — und Nutzungsdaten beantworten genau diese Frage.

- **Sammeln Sie tatsächliche Nutzung** über einen Zeitraum, der den vollständigen Geschäftszyklus abdeckt, einschließlich Periodenende- und Jahresprozessen. Alles Kürzere klassifiziert eine legitim seltene Berechtigung als ungenutzt, und ein solcher Fehler wird das Programm stoppen.
- **Vergleichen Sie Gewährtes gegen Genutztes pro Nutzer**, und behandeln Sie die Differenz als das Working Set. Dieser Vergleich allein etabliert üblicherweise, dass eine große Mehrheit zugewiesener Berechtigungen nie von der sie haltenden Person ausgeübt wurde.
- **Leiten Sie Kandidatenrollen aus Clustern tatsächlicher Nutzung ab**, dann gleichen Sie diese Cluster mit der Geschäftsseite gegen Funktionsbezeichnungen ab. Rein datenabgeleitete Rollen sind unwartbar; rein aus Stellenbeschreibungen abgeleitete Rollen passen nicht zu dem, was Menschen tun. Der Abgleich ist die Arbeit.
- **Behandeln Sie Ausnahmen explizit**, statt eine Rolle zu erweitern, um sie unterzubringen. Ein Nutzer, der etwas außerhalb seiner Rolle braucht, sollte eine separate, zeitlich begrenzte Gewährung mit einer Aufzeichnung erhalten, was das Modell kohärent hält.
- **Entfernen Sie stufenweise mit einer überwachten Periode.** Ziehen Sie die Berechtigung zurück, aber protokollieren Sie den Versuch, der sie genutzt hätte, dann entfernen Sie sie tatsächlich, sobald ein Zyklus ohne Treffer vergeht. Dies verwandelt Entfernung von einem Glücksspiel in eine Messung.
- **Etablieren Sie den Entfernungsprozess vor Abschluss der Bereinigung**: einen Verlassensprozess, der widerruft, einen Wechselprozess, der das Alte entfernt ebenso wie das Neue hinzufügt, und einen periodischen Review mit einem benannten Eigentümer pro Rolle. Ohne diese ist die Bereinigung eine Einmalaktion.
- **Geben Sie jeder Rolle einen Eigentümer und einen festgelegten Zweck.** Eine eigentümerlose Rolle kann nicht überprüft werden, und die Abwesenheit eines Eigentümers ist die Bedingung, unter der Rollen permanent werden.
- **Erzwingen Sie Aufgabentrennungsregeln im Modell**, nicht als periodischer Audit-Befund. Zum Zuweisungszeitpunkt erkannte Konflikte werden verhindert; jährlich erkannte Konflikte werden gemeldet.
- **Berichten Sie die Größe des Modells als verfolgte Kennzahl**, sodass erneutes Wachstum früh sichtbar wird, statt beim nächsten Audit entdeckt zu werden.

## Tradeoffs ⇄

> Bereinigung stellt ein Modell wieder her, das beschrieben und auditiert werden kann, aber die Entfernung von Zugriff trägt echtes Risiko, legitime Arbeit zu blockieren, und der Aufwand ist erheblich.

**Vorteile:**

- Die Frage, wer eine sensible Aktion durchführen kann, wird beantwortbar, was die Fähigkeit ist, die sowohl Sicherheit als auch Audit tatsächlich brauchen.
- Übermäßiger Zugriff wird entfernt, was die Exposition durch ein kompromittiertes Konto und durch Insider-Missbrauch reduziert.
- Provisionierung wird schneller und konsistenter, weil eine definierte Rolle zum Zuweisen existiert statt der Satz eines Kollegen zum Kopieren.
- Zugriffs-Reviews werden bedeutsam, da Manager nach einer verständlichen Rolle statt einer Liste technischer Berechtigungen gefragt werden.
- Der Entfernungsprozess stoppt das erneute Wachstum, was dies von den periodischen Bereinigungen unterscheidet, die die meisten Organisationen bereits versucht haben.

**Kosten und Risiken:**

- Die Entfernung einer Berechtigung, die sich als benötigt herausstellt, blockiert die Arbeit von jemandem, und ein paar sichtbare Fälle können das Programm beenden.
- Nutzungsdaten über einen unzureichenden Zeitraum klassifizieren systematisch seltene, aber essenzielle Berechtigungen falsch, und die seltensten sind oft die kritischsten.
- Der Abgleich zwischen Nutzungsclustern und Funktionsbezeichnungen ist langsam, politisch und erfordert Geschäftsbeteiligung, die schwer zu erhalten ist.
- Bereinigung ohne Prozessänderung ist vorübergehend, und der Aufwand muss innerhalb weniger Jahre wiederholt werden.
- Manche Systeme zeichnen Nutzung schlecht oder gar nicht auf, in welchem Fall die evidentielle Grundlage schwach ist und die Arbeit weit riskanter wird.

## How It Could Be

Eine ERP-Installation hatte 3.100 Rollen für 2.400 Nutzer. Nutzung wurde dreizehn Monate lang gesammelt, um den Jahresabschluss abzudecken. Der Vergleich zeigte, dass über alle Nutzer hinweg ungefähr 71 Prozent der gewährten Berechtigungen nie vom Halter ausgeübt worden waren. Das Clustering tatsächlicher Nutzung produzierte 140 Kandidatenrollen, was der Abgleich mit der Geschäftsseite in 190 verwandelte — die zusätzlichen 50 deckten legitime Varianten ab, die die Daten zusammengefasst hatten. Entfernung wurde stufenweise durchgeführt: Berechtigungen wurden zurückgezogen, aber Versuche protokolliert, und über den folgenden Zyklus identifizierten 340 protokollierte Versuche echt benötigten Zugriff, den das Nutzungsfenster übersehen hatte, alles davon wurde wiederhergestellt, bevor irgendein Nutzer blockiert wurde. Das Modell endete bei 190 Rollen mit jeweils einem Eigentümer.

Die Prozessänderung war es, die es haltbar machte, und es war der Teil, den die Organisation in zwei vorherigen Bereinigungen übersprungen hatte. Ein Wechselprozess, der alten Zugriff entfernte ebenso wie neuen gewährte, ein Verlassensprozess, der widerrief, vierteljährlicher Review durch Rolleneigentümer statt durch Linienmanager, und eine verfolgte Zahl, berichtet neben anderen betrieblichen Kennzahlen. Zwei Jahre später stand das Modell bei 210 Rollen statt den mehreren Tausend, die es nach jeder vorherigen Bereinigung erreicht hatte. Die Einschätzung des Sicherheitsteams war, dass die früheren Versuche nicht gescheitert waren, weil die Bereinigung falsch war, sondern weil sich nichts daran geändert hatte, wie Zugriff am Tag nach ihrem Abschluss gewährt wurde.
