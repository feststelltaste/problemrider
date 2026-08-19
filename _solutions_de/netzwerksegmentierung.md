---
title: Netzwerksegmentierung
description: Aufteilung des Netzwerks in Sicherheitszonen mit getrennten
  Vertrauensstufen.
category:
- Security
- Architecture
problems:
- cascade-failures
- insecure-data-transmission
- data-protection-risk
- system-outages
- authorization-flaws
- monitoring-gaps
- poor-system-environment
layout: solution
lang: de
en_slug: network-segmentation
related_solutions:
- slug: defense-lines
  similarity: 0.8
- slug: honeypots
  similarity: 0.75
- slug: trust-boundaries
  similarity: 0.75
- slug: patch-management
  similarity: 0.75
- slug: incident-response-measures
  similarity: 0.75
- slug: least-privilege
  similarity: 0.75
---

## Description

Netzwerksegmentierung teilt ein Netzwerk in unterschiedliche Zonen — jede mit ihrer eigenen Vertrauensstufe und ihrer eigenen Menge erlaubter Kommunikationspfade —, sodass eine Kompromittierung in einer Zone einem Angreifer oder einem sich ausbreitenden Ausfall nicht automatisch Zugriff auf jede andere Komponente gewährt. Sie wird durch Firewalls, Netzwerkrichtlinien oder softwaredefinierte Grenzen umgesetzt, die eine Default-Deny-Haltung durchsetzen: Nur explizit erlaubte Ports, Protokolle und Quelle/Ziel-Paare dürfen eine Zonengrenze überschreiten, und alles andere wird blockiert und protokolliert. Legacy-Umgebungen sind erstklassige Kandidaten für diese Behandlung, weil sie häufig auf flachen Netzwerken gewachsen sind, wo jeder Server jeden anderen Server direkt erreichen konnte, ein Muster, das damals Ad-hoc-Integration erleichterte, aber keine interne Barriere hinterlässt, sobald Perimeter-Verteidigungen umgangen werden. Dies ist besonders folgenreich für Legacy-Komponenten, die aus Kompatibilitätsgründen nicht gepatcht oder aktualisiert werden können: Segmentierung erlaubt, solche Systeme hinter strikten, überwachten Grenzen als kompensierende Kontrolle zu isolieren und ihr Risiko einzudämmen, ohne die zugrundeliegende Schwachstelle beheben zu müssen. Segmentierung begrenzt auch den Explosionsradius nicht-sicherheitsbezogener Vorfälle wie Malware-Ausbrüche oder kaskadierender Ausfälle, da ein auf ein Segment beschränkter Fehler sich nicht über uneingeschränkte Netzwerkpfade zu unzusammenhängenden Systemen ausbreiten kann. Die Hauptkosten sind, dass genau die Abhängigkeiten, die Segmentierung kontrollieren soll, in Legacy-Umgebungen oft undokumentiert sind, sodass die Einführung strikter Grenzen sorgfältige Entdeckung echter Verkehrsmuster erfordert, um funktionierende Integrationen nicht zu brechen.

## How to Apply ◆

> Legacy-Systeme operieren häufig auf flachen Netzwerken, wo jede kompromittierte Komponente jede andere erreichen kann. Netzwerksegmentierung teilt das Netzwerk in Zonen mit unterschiedlichen Vertrauensstufen auf, begrenzt laterale Bewegung und dämmt den Explosionsradius von Kompromittierungen ein.

- Kartieren Sie alle Netzwerkkommunikationspfade zwischen Legacy-Systemkomponenten und identifizieren Sie, welche Verbindungen tatsächlich erforderlich sind, damit das System funktioniert. Viele Legacy-Systeme haben offenen Netzwerkzugriff zwischen Komponenten, die nicht direkt kommunizieren müssen.
- Definieren Sie Sicherheitszonen basierend auf Datensensitivität und Vertrauensstufe: DMZ für internetzugewandte Komponenten, Anwendungszone für Geschäftslogik, Datenzone für Datenbanken, Verwaltungszone für administrativen Zugriff und isolierte Zonen für Legacy-Komponenten mit bekannten Schwachstellen.
- Implementieren Sie Firewall-Regeln oder Netzwerkrichtlinien zwischen Zonen, die nur die spezifischen Ports, Protokolle und Quelle/Ziel-Paare erlauben, die für legitime Kommunikation erforderlich sind. Default-Deny-Regeln stellen sicher, dass jeder neue, nicht genehmigte Kommunikationspfad blockiert wird.
- Platzieren Sie Legacy-Systeme mit bekannten, ungepatchten Schwachstellen in isolierten Segmenten mit strikt begrenztem eingehendem und ausgehendem Zugriff. Diese Segmente sollten verstärktes Monitoring haben, um Ausnutzungsversuche zu erkennen.
- Implementieren Sie Mikrosegmentierung für kritische Komponenten: einzelne Datenbankserver, Zahlungsverarbeitungssysteme und administrative Schnittstellen sollten ihre eigenen Netzwerkrichtlinien haben, selbst innerhalb einer breiteren Zone.
- Setzen Sie Netzwerküberwachung an Zonengrenzen ein, um unautorisierte Kommunikationsversuche zu erkennen. Jeglicher Verkehr, der eine Deny-Regel trifft, stellt entweder eine Fehlkonfiguration oder einen Angriffsversuch dar und sollte einen Alarm erzeugen.
- Dokumentieren Sie die Netzwerkarchitektur und Segmentierungsregeln, damit Infrastrukturänderungen die beabsichtigten Sicherheitsgrenzen aufrechterhalten, statt unbeabsichtigt neue Pfade durch Zonen zu schaffen.

## Tradeoffs ⇄

> Netzwerksegmentierung dämmt den Explosionsradius von Kompromittierungen ein und begrenzt laterale Bewegung, fügt aber Netzwerkkomplexität hinzu und kann Legacy-Systemkommunikationsmuster beeinflussen.

**Vorteile:**

- Begrenzt die Auswirkung eines Sicherheitsverstoßes, indem der Angreifer auf das kompromittierte Segment beschränkt wird, statt Zugriff auf das gesamte Netzwerk zu gewähren.
- Bietet kompensierende Kontrollen für Legacy-Systeme, die nicht gepatcht werden können, indem sie von potenziellen Angriffsvektoren isoliert werden.
- Ermöglicht unterschiedliche Sicherheitsüberwachungsintensitäten für verschiedene Zonen basierend auf ihrer Sensitivität und ihrem Risikoniveau.
- Unterstützt regulatorische Compliance, indem demonstriert wird, dass sensible Datenumgebungen von Allzweck-Netzwerken isoliert sind.

**Kosten und Risiken:**

- Legacy-Systeme könnten undokumentierte Netzwerkabhängigkeiten haben, die brechen, wenn Segmentierung implementiert wird, was sorgfältige Entdeckung und Testen erfordert.
- Netzwerksegmentierung fügt der Infrastrukturverwaltung betriebliche Komplexität hinzu und erfordert sorgfältige Änderungskontrolle, um Segmentierungsregeln aufrechtzuerhalten.
- Die Performance könnte beeinträchtigt werden, wenn Segmentierung zusätzliche Netzwerk-Hops oder Firewall-Verarbeitung für hochvolumige Interkomponentenkommunikation einführt.
- Unsachgemäß implementierte Segmentierung kann ein falsches Sicherheitsgefühl erzeugen, wenn Ausnahmen und Umgehungsregeln zu freizügig sind.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Netzwerksegmentierung Legacy-Systeme schützt.

Eine Legacy-Webanwendung, ihr Datenbankserver und ein Dateiserver befinden sich alle auf demselben flachen Netzwerksegment. Ein Angreifer nutzt eine Schwachstelle in der Webanwendung aus und entdeckt sofort, dass er sich direkt mit dem Datenbankserver auf dessen Standardport verbinden kann, wobei er die Zugriffskontrollen der Anwendung vollständig umgeht. Nach der Implementierung von Netzwerksegmentierung wird die Webanwendung in einer Anwendungszone platziert, die Datenbank in einer Datenzone und der Dateiserver in einer Speicherzone. Firewall-Regeln erlauben nur dem Anwendungsserver, sich mit der Datenbank auf dem spezifischen von der Anwendung genutzten Port zu verbinden, und nur der Anwendungsserver kann auf den Dateiserver zugreifen. Als eine nachfolgende Schwachstelle in der Webanwendung ausgenutzt wird, stellt der Angreifer fest, dass er die Datenbank nicht mehr direkt erreichen kann — die Firewall blockiert alle nicht genehmigten Verbindungen, und der Versuch löst einen Netzwerküberwachungsalarm aus, der Vorfallreaktion einleitet.

Ein Unternehmen betreibt einen Legacy-Mainframe, der Finanztransaktionen neben modernen Webdiensten im selben Unternehmensnetzwerk verarbeitet. Ein Ransomware-Ausbruch, der über eine Phishing-E-Mail auf einer Unternehmensarbeitsstation eintritt, breitet sich über das flache Netzwerk aus und erreicht schließlich die Netzwerkschnittstelle des Mainframes. Die Batch-Verarbeitung des Mainframes hält für 48 Stunden an, während die Ransomware eingedämmt wird. Nach der Wiederherstellung implementiert das Team Netzwerksegmentierung, die den Mainframe in eine isolierte Hochsicherheitszone platziert. Nur zwei spezifische Anwendungsserver dürfen mit dem Mainframe kommunizieren, und nur über die spezifischen Ports, die für Transaktionseinreichung und Ergebnisabruf genutzt werden. Aller anderer Netzwerkverkehr zur Mainframe-Zone wird blockiert und protokolliert. Als ein nachfolgender Ransomware-Vorfall das Unternehmensnetzwerk betrifft, bleibt die Mainframe-Zone völlig unbetroffen, weil kein Kommunikationspfad vom infizierten Segment zur Mainframe-Zone existiert.
